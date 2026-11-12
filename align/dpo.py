"""Direct Preference Optimisation (Rafailov et al. 2023), written from scratch.

DPO's trick is that you never train a separate reward model or run PPO rollouts: the
optimal policy for a KL-constrained reward maximisation problem has a closed form, and
substituting it into the Bradley-Terry preference model turns "maximise reward" into a
plain classification loss between a trainable *policy* and a frozen *reference* copy of
the same model::

    L(theta) = -log sigmoid( beta * [ (logp_policy(chosen) - logp_ref(chosen))
                                      - (logp_policy(rejected) - logp_ref(rejected)) ] )

``beta`` sets how far the policy is allowed to drift from the reference before the loss
stops caring about the preference and starts caring about the KL term implicitly baked
into that subtraction. Low beta -> larger drift, more constraint satisfaction, more of the
reward hacking this repo is built to show; high beta -> safer, smaller gains. The beta
sweep in ``configs/dpo-beta-sweep.json`` trains several of these side by side on purpose.

Usage::

    python -m align.dpo configs/dpo-beta0.1.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from align import model_utils


@dataclass
class DPOConfig:
    run_name: str = "dpo-beta0.1"
    pairs_path: str = "data/pairs_train.jsonl"
    checkpoint: str = str(model_utils.DEFAULT_CHECKPOINT)
    tokenizer: str = str(model_utils.DEFAULT_TOKENIZER)
    beta: float = 0.1
    lr: float = 1e-5
    epochs: int = 3
    grad_clip: float = 1.0
    log_interval: int = 5
    ckpt_interval: int = 20
    out_dir: str = "runs"

    @classmethod
    def from_file(cls, path: str | Path) -> DPOConfig:
        return cls(**json.loads(Path(path).read_text()))


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss, implicit_reward_margin, accuracy) for one pair."""
    policy_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)
    margin = logits.detach() / beta  # the implicit reward margin, in nats, beta-free
    acc = (logits > 0).float()
    return loss, margin, acc


def load_pairs(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def train(cfg: DPOConfig) -> Path:
    tokenizer = model_utils.load_tokenizer(cfg.tokenizer)
    policy = model_utils.load_model(cfg.checkpoint)
    for p in policy.parameters():
        p.requires_grad_(True)
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    device = next(policy.parameters()).device
    device_type = "cuda" if device.type == "cuda" else "cpu"
    optim = policy.configure_optimizer(cfg.lr, weight_decay=0.0, betas=(0.9, 0.95), device_type=device_type)

    pairs = load_pairs(cfg.pairs_path)
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")

    step = 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        for pair in pairs:
            prompt_ids = tokenizer.encode(pair["prompt"])
            chosen_ids = tokenizer.encode(pair["chosen"])
            rejected_ids = tokenizer.encode(pair["rejected"])

            policy_chosen = model_utils.sequence_logprobs(policy, prompt_ids, chosen_ids)
            policy_rejected = model_utils.sequence_logprobs(policy, prompt_ids, rejected_ids)
            with torch.no_grad():
                ref_chosen = model_utils.sequence_logprobs(reference, prompt_ids, chosen_ids)
                ref_rejected = model_utils.sequence_logprobs(reference, prompt_ids, rejected_ids)

            loss, margin, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, cfg.beta)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optim.step()

            if step % cfg.log_interval == 0:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "reward_margin": float(margin.item()),
                    "acc": float(acc.item()),
                    "kl_chosen": float((policy_chosen - ref_chosen).item()),
                    "elapsed": time.perf_counter() - t0,
                }
                with metrics_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[dpo:{cfg.run_name}] step {step:4d} loss {rec['loss']:.4f} "
                      f"margin {rec['reward_margin']:+.3f} acc {rec['acc']:.2f}")

            if step % cfg.ckpt_interval == 0 and step > 0:
                _save(policy, cfg, out_dir / f"step{step}.pt", step)
            step += 1

    final = out_dir / "best.pt"
    _save(policy, cfg, final, step)
    print(f"[dpo:{cfg.run_name}] done: {step} steps, {time.perf_counter() - t0:.0f}s -> {final}")
    return final


def _save(model, cfg: DPOConfig, path: Path, step: int) -> None:
    """``step`` is stored in the checkpoint itself -- evaluate.py reads it back rather than
    guessing from the filename, so ``best.pt`` (the final checkpoint) is never confused with
    step 0 (the untouched base model, which is evaluated separately as its own reference row).
    """
    state = {"model": model.state_dict(), "config": asdict(model.cfg), "dpo_config": asdict(cfg), "step": step}
    torch.save(state, path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args(argv)
    train(DPOConfig.from_file(args.config))


if __name__ == "__main__":
    main()
