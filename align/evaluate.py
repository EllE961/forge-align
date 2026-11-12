"""Evaluate trained DPO checkpoints on held-out tasks against two numbers:

* **proxy reward** -- ``align.reward.score``, the exact signal DPO was trained on.
* **true fluency** -- mean log-probability per token of the generated completion under
  the *original, untouched* base checkpoint. This was never part of the training reward,
  so if it falls while proxy reward rises, that is reward hacking, measured, not asserted.

Usage::

    python -m align.evaluate --run dpo-beta0.1          # one run, all its saved steps
    python -m align.evaluate --sweep                     # final checkpoint of every run in configs/dpo-beta-sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from align import model_utils, reward
from align.prompts import split_tasks


def evaluate_checkpoint(
    ckpt_path: str | Path, base_model, tokenizer, tasks, samples_per_task: int = 3, step: int | None = None
) -> dict:
    policy = model_utils.load_model(ckpt_path)
    if step is None:
        # trained checkpoints carry their own step (see align.dpo._save); the base checkpoint
        # does not, and is always step 0 -- the untouched reference everything else is judged against.
        step = policy.checkpoint_meta.get("step", 0)
    proxy_scores, fluency_scores = [], []
    examples = []
    for ti, task in enumerate(tasks):
        for s in range(samples_per_task):
            text = model_utils.sample_text(policy, tokenizer, task.prompt, seed=10_000 + ti * samples_per_task + s)
            rb = reward.score(text, task.keyword, task.tone)
            comp_ids = tokenizer.encode(text)[len(tokenizer.encode(task.prompt)):]
            fluency = model_utils.mean_logprob_per_token(base_model, tokenizer.encode(task.prompt), comp_ids)
            proxy_scores.append(rb.total)
            fluency_scores.append(fluency)
            if s == 0:
                examples.append(
                    {"task": task.keyword, "tone": task.tone, "text": text, "reward": rb.total, "fluency": fluency}
                )
    n = len(proxy_scores)
    return {
        "checkpoint": str(ckpt_path),
        "step": step,
        "n": n,
        "proxy_reward_mean": sum(proxy_scores) / n,
        "true_fluency_mean": sum(fluency_scores) / n,
        "examples": examples,
    }


def evaluate_run(run_name: str, runs_dir: str = "runs", out_dir: str = "results", samples_per_task: int = 3) -> Path:
    tokenizer = model_utils.load_tokenizer()
    base_model = model_utils.load_model(model_utils.DEFAULT_CHECKPOINT)
    _, eval_tasks = split_tasks()

    run_dir = Path(runs_dir) / run_name
    ckpts = sorted(run_dir.glob("step*.pt"), key=lambda p: int(p.stem.replace("step", "")))
    ckpts.append(run_dir / "best.pt")

    rows = []
    base_eval = evaluate_checkpoint(
        model_utils.DEFAULT_CHECKPOINT, base_model, tokenizer, eval_tasks, samples_per_task, step=0
    )
    rows.append(base_eval)
    print(f"[evaluate] {run_name} step {base_eval['step']:>5} (base) proxy {base_eval['proxy_reward_mean']:.3f} "
          f"fluency {base_eval['true_fluency_mean']:.3f}")
    for ckpt in ckpts:
        if not ckpt.exists():
            continue
        row = evaluate_checkpoint(ckpt, base_model, tokenizer, eval_tasks, samples_per_task)
        rows.append(row)
        print(f"[evaluate] {run_name} step {row['step']:>5} proxy {row['proxy_reward_mean']:.3f} "
              f"fluency {row['true_fluency_mean']:.3f}")

    out = Path(out_dir) / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run name under runs/, e.g. dpo-beta0.1")
    ap.add_argument("--sweep", action="store_true", help="evaluate every run listed in configs/dpo-beta-sweep.json")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args(argv)

    if args.sweep:
        sweep = json.loads(Path("configs/dpo-beta-sweep.json").read_text())
        for run_name in sweep["runs"]:
            evaluate_run(run_name, samples_per_task=args.samples)
    elif args.run:
        evaluate_run(args.run, samples_per_task=args.samples)
    else:
        ap.error("pass --run NAME or --sweep")


if __name__ == "__main__":
    main()
