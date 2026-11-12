"""Playground backend (FastAPI). Serves the single-page UI in ``web/`` over one API:

    GET  /api/status             which models (base + each trained beta) are available
    POST /api/compare            base vs. every aligned model's completion + reward breakdown,
                                  for one (keyword, tone) task, plus the per-token shift for beta=0.1

Run::

    python -m align.server --port 8002
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from align import model_utils, reward
from align.prompts import TONES, build_prompt

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
RUNS_DIR = ROOT / "runs"
mimetypes.add_type("application/javascript", ".js")

app = FastAPI(title="forge-align playground")

_tokenizer = None
_base_model = None
_aligned_cache: dict[str, object] = {}


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = model_utils.load_tokenizer()
    return _tokenizer


def base_model():
    global _base_model
    if _base_model is None:
        _base_model = model_utils.load_model(model_utils.DEFAULT_CHECKPOINT)
    return _base_model


def aligned_model(run_name: str):
    if run_name not in _aligned_cache:
        _aligned_cache[run_name] = model_utils.load_model(RUNS_DIR / run_name / "best.pt")
    return _aligned_cache[run_name]


def available_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "best.pt").exists())


class CompareRequest(BaseModel):
    keyword: str = "dragon"
    tone: str = "happy"
    seed: int = 0


@app.get("/api/status")
def status() -> dict:
    return {
        "base_checkpoint": str(model_utils.DEFAULT_CHECKPOINT),
        "runs": available_runs(),
        "tones": TONES,
    }


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    tok = tokenizer()
    prompt = build_prompt(req.keyword, req.tone)
    prompt_ids = tok.encode(prompt)

    def run_one(model) -> dict:
        text = model_utils.sample_text(model, tok, prompt, seed=req.seed)
        rb = reward.score(text, req.keyword, req.tone)
        return {"text": text, "reward": rb.__dict__}

    out = {"prompt": prompt, "base": run_one(base_model())}
    for run_name in available_runs():
        out[run_name] = run_one(aligned_model(run_name))

    # per-token probability shift for the first available aligned run, vs. the base model
    runs = available_runs()
    if runs:
        aligned = aligned_model(runs[0])
        comp_text = out[runs[0]]["text"]
        comp_ids = tok.encode(comp_text)[len(prompt_ids):]
        if comp_ids:
            with torch.no_grad():
                ref_logp = _per_token_logp(base_model(), prompt_ids, comp_ids)
                pol_logp = _per_token_logp(aligned, prompt_ids, comp_ids)
            out["token_shift"] = {
                "run": runs[0],
                "tokens": [tok.decode([t]) for t in comp_ids],
                "ref_logp": ref_logp,
                "policy_logp": pol_logp,
            }
    return out


def _per_token_logp(model, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids + completion_ids, dtype=torch.long, device=device)[None]
    logits, _ = model(ids[:, :-1])
    logp = torch.log_softmax(logits.float(), dim=-1)
    targets = ids[:, 1:]
    tok_logp = logp.gather(-1, targets[..., None]).squeeze(-1)[0]
    start = len(prompt_ids) - 1
    return tok_logp[start:].tolist()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
