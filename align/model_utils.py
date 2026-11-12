"""Thin glue over forge-llm: load checkpoints, sample completions, score log-likelihood.

Nothing here reimplements the tokenizer or the transformer -- both come straight from
``forge`` (forge-llm). This module only adds the two things alignment needs that plain
inference does not: per-sequence log-probabilities of a *given* completion (for the DPO
loss) and a convenience sampler that returns decoded text instead of token ids.
"""

from __future__ import annotations

from pathlib import Path

import torch
from forge.generate import generate
from forge.model import Transformer
from forge.tokenizer import Tokenizer

FORGE_LLM_DIR = Path(__file__).resolve().parents[2] / "forge-llm"
DEFAULT_CHECKPOINT = FORGE_LLM_DIR / "checkpoints" / "tinystories-30m" / "best.pt"
DEFAULT_TOKENIZER = FORGE_LLM_DIR / "data" / "tokenizer.json"


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER) -> Tokenizer:
    return Tokenizer.load(path)


def load_model(
    path: str | Path = DEFAULT_CHECKPOINT, device: str = "cuda", dtype: torch.dtype = torch.float32
) -> Transformer:
    device = device if torch.cuda.is_available() else "cpu"
    model = Transformer.from_checkpoint(path, device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def sample_text(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 90,
    temperature: float = 0.9,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    seed: int | None = None,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    return tokenizer.decode(out_ids)


def sequence_logprobs(model: Transformer, prompt_ids: list[int], completion_ids: list[int]) -> torch.Tensor:
    """Sum of log p(completion_t | prompt, completion_<t>) under ``model``, teacher-forced.

    Returns a 0-d tensor that keeps gradients (the caller decides whether to call this
    under ``torch.no_grad()``, since the policy needs gradients and the reference does not).
    """
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + completion_ids, dtype=torch.long, device=device)[None]
    logits, _ = model(ids[:, :-1])
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]
    tok_logp = log_probs.gather(-1, targets[..., None]).squeeze(-1)  # (1, T-1)
    start = len(prompt_ids) - 1  # index into tok_logp of the first completion token's logp
    return tok_logp[0, start:].sum()


def mean_logprob_per_token(model: Transformer, prompt_ids: list[int], completion_ids: list[int]) -> float:
    """Average (not summed) log-prob per completion token -- the length-independent fluency score."""
    if not completion_ids:
        return 0.0
    with torch.no_grad():
        total = sequence_logprobs(model, prompt_ids, completion_ids)
    return float(total.item()) / len(completion_ids)
