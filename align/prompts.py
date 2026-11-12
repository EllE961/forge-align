"""Constrained-generation tasks: "write a story about X that ends Y."

Forge's base model is a plain next-token predictor trained on TinyStories, not an
instruction-tuned model -- it was never taught to *follow* an instruction, only to
continue one as text. That's deliberate here: it means the base model satisfies these
constraints only by chance, which gives DPO something real to improve, and it means the
"alignment" in this repo is teaching instruction-following itself, not just a safety
filter on top of an already-compliant model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

KEYWORDS = [
    "dragon",
    "puppy",
    "garden",
    "kite",
    "robot",
    "forest",
    "birthday",
    "ocean",
    "star",
    "cookie",
    "balloon",
    "rabbit",
    "castle",
    "train",
    "snow",
    "butterfly",
    "treehouse",
    "pirate",
    "rainbow",
    "turtle",
]
TONES = ["happy", "sad"]

TEMPLATE = (
    "Write a short story about a {keyword}. The story should end {tone}.\n\nOnce upon a time,"
)


@dataclass(frozen=True)
class Task:
    keyword: str
    tone: str
    prompt: str


def build_prompt(keyword: str, tone: str) -> str:
    return TEMPLATE.format(keyword=keyword, tone=tone)


def all_tasks() -> list[Task]:
    return [Task(k, t, build_prompt(k, t)) for k in KEYWORDS for t in TONES]


def split_tasks(seed: int = 0, train_frac: float = 0.7) -> tuple[list[Task], list[Task]]:
    """Fixed, seeded split so train/eval tasks never overlap and runs are reproducible."""
    tasks = all_tasks()
    rng = random.Random(seed)
    rng.shuffle(tasks)
    cut = int(len(tasks) * train_frac)
    return tasks[:cut], tasks[cut:]
