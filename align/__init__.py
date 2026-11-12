"""forge-align: DPO alignment for the Forge language model.

This package does not reimplement a tokenizer, transformer or sampler -- it imports those
from ``forge`` (the forge-llm package) and adds everything alignment-specific on top:
a programmatic reward model, preference-pair generation, a from-scratch DPO trainer,
an evaluation harness that looks for reward hacking, and the visualisations in the README.
"""

from __future__ import annotations

__all__ = ["reward", "prompts", "preference_data", "dpo", "evaluate", "visualize"]
