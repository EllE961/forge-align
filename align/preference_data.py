"""Build preference pairs by sampling K completions per task from the base model and
scoring every one with the proxy reward (align.reward) -- no human labelling, no external
judge. The best-scoring completion becomes "chosen", the worst becomes "rejected".

Usage::

    python -m align.preference_data --split train --k 8 --out data/pairs_train.jsonl
    python -m align.preference_data --split eval  --k 8 --out data/pairs_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from align import model_utils, reward
from align.prompts import Task, split_tasks


def sample_and_score(model, tokenizer, task: Task, k: int, seed_base: int) -> list[dict]:
    out = []
    for i in range(k):
        text = model_utils.sample_text(model, tokenizer, task.prompt, seed=seed_base + i)
        rb = reward.score(text, task.keyword, task.tone)
        out.append({"text": text, "reward": rb.total, "breakdown": rb.__dict__})
    return out


def build_pairs(tasks: list[Task], model, tokenizer, k: int, seed_base: int) -> list[dict]:
    pairs = []
    for ti, task in enumerate(tasks):
        samples = sample_and_score(model, tokenizer, task, k, seed_base=seed_base + ti * k)
        samples.sort(key=lambda s: s["reward"])
        worst, best = samples[0], samples[-1]
        if best["reward"] - worst["reward"] < 1e-6:
            continue  # no signal in this batch -- skip rather than train on a tie
        pairs.append(
            {
                "prompt": task.prompt,
                "keyword": task.keyword,
                "tone": task.tone,
                "chosen": best["text"],
                "rejected": worst["text"],
                "chosen_reward": best["reward"],
                "rejected_reward": worst["reward"],
            }
        )
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--k", type=int, default=8, help="completions sampled per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=str(model_utils.DEFAULT_CHECKPOINT))
    ap.add_argument("--tokenizer", default=str(model_utils.DEFAULT_TOKENIZER))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    train_tasks, eval_tasks = split_tasks(seed=args.seed)
    tasks = train_tasks if args.split == "train" else eval_tasks
    out_path = Path(args.out or f"data/pairs_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = model_utils.load_tokenizer(args.tokenizer)
    model = model_utils.load_model(args.checkpoint)

    pairs = build_pairs(tasks, model, tokenizer, k=args.k, seed_base=args.seed * 1000)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[preference_data] {len(pairs)}/{len(tasks)} tasks produced a usable pair -> {out_path}")


if __name__ == "__main__":
    main()
