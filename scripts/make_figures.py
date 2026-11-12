"""Render the README figures from results/*.json.

Same visual system as forge-llm's figures on purpose (soot background, hairline grid,
the validated three-colour palette) -- these two repos are read as one body of work.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("docs/assets")
RES = Path("results")

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE = "#d95926", "#3987e5", "#c0509c"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "xtick.major.pad": 6, "ytick.major.pad": 6, "axes.axisbelow": True,
    "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})


def overfitting_trajectory(run_json: Path, out: Path, beta: float) -> None:
    """Not the textbook reward-hacking curve (proxy up, fluency down) -- at tight beta, both
    fall together. That is overfitting 28 examples, a different and equally real failure mode,
    and it is what the data at this beta actually shows, so that is what the chart says.
    """
    rows = json.loads(run_json.read_text())
    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    proxy = [r["proxy_reward_mean"] for r in rows]
    fluency = [r["true_fluency_mean"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax2.tick_params(colors=INK3)
    l1, = ax1.plot(steps, proxy, color=EMBER, marker="o", label="proxy reward (the training signal)")
    l2, = ax2.plot(steps, fluency, color=TEMPER, marker="o", label="true fluency (held out, never trained on)")
    ax1.axhline(proxy[0], color=EMBER, lw=1, ls="--", alpha=0.5)
    ax1.annotate("base model's proxy reward", (steps[0], proxy[0]), xytext=(6, 6),
                 textcoords="offset points", color=EMBER, fontsize=8.5, alpha=0.8)
    ax1.set_xlabel("DPO step")
    ax1.set_ylabel("proxy reward", color=EMBER)
    ax2.set_ylabel("fluency (mean log p / token)", color=TEMPER)
    ax1.set_title(f"β={beta}: both fall together -- overfitting 28 pairs, not reward hacking")
    # mark the step where the sustained decline begins (the last local peak in proxy, not just
    # the single biggest one-step drop -- a true turning point, not a noise spike)
    peaks = [i for i in range(1, len(proxy) - 1) if proxy[i] >= proxy[i - 1] and proxy[i] >= proxy[i + 1]]
    turn = peaks[-1] if peaks else 0
    ax1.annotate("proxy peaks, then neither recovers", xy=(steps[turn], proxy[turn]),
                 xytext=(10, -22), textcoords="offset points", color=INK2, fontsize=9,
                 arrowprops={"arrowstyle": "-", "color": INK3, "lw": 1})
    ax1.legend(handles=[l1, l2], loc="center right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def training_dynamics(metrics_jsonl: Path, out: Path, title_suffix: str = "") -> None:
    rows = [json.loads(line) for line in metrics_jsonl.read_text().splitlines() if line.strip()]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, label, color in [
        (axes[0], "loss", "DPO loss", EMBER),
        (axes[1], "reward_margin", "implicit reward margin", TEMPER),
        (axes[2], "acc", "pairwise accuracy", PURPLE),
    ]:
        ax.plot(steps, [r[key] for r in rows], color=color)
        ax.set_title(label)
        ax.set_xlabel("step")
    fig.suptitle(f"DPO training dynamics{title_suffix}", color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)


def beta_sweep(result_paths: list[Path], betas: list[float], out: Path) -> None:
    finals = []
    for p in result_paths:
        rows = json.loads(p.read_text())
        finals.append(max(rows, key=lambda r: r["step"]))  # the last/highest-step checkpoint, i.e. "best.pt"
    proxy = [r["proxy_reward_mean"] for r in finals]
    fluency = [r["true_fluency_mean"] for r in finals]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(proxy, fluency, color=INK3, lw=1.2, zorder=1)
    ax.scatter(proxy, fluency, s=90, c=[EMBER, TEMPER, PURPLE, "#3bb273"][: len(proxy)], zorder=2)
    for b, x, y in zip(betas, proxy, fluency):
        ax.annotate(f"β={b}", (x, y), xytext=(8, 6), textcoords="offset points", color=INK2, fontsize=10)
    ax.set_xlabel("proxy reward (constraint satisfaction)")
    ax.set_ylabel("true fluency (held out)")
    ax.set_title("The alignment tax: constraint satisfaction vs. fluency, by β")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def token_shift(
    policy_tokens: list[str], ref_logp: list[float], policy_logp: list[float], out: Path,
    tokens_per_row: int = 14, max_tokens: int = 56,
) -> None:
    """A per-token strip, colour-coded by how much more/less likely each token became after DPO.

    Wrapped into fixed-width rows (like a diff view) instead of one long strip -- a strip wide
    enough for 100+ tokens is unreadable at any sane figure size, wrapped rows are not.
    """
    tokens = policy_tokens[:max_tokens]
    diff = (np.array(policy_logp) - np.array(ref_logp))[:max_tokens]
    vmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
    n_rows = -(-len(tokens) // tokens_per_row)  # ceil

    fig, ax = plt.subplots(figsize=(11, 0.62 * n_rows + 0.5))
    ax.set_xlim(0, tokens_per_row)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.axis("off")
    for i, (tok, d) in enumerate(zip(tokens, diff)):
        row, col = divmod(i, tokens_per_row)
        t = (d / vmax + 1) / 2  # -> [0, 1]
        color = plt.cm.RdBu_r(t)  # type: ignore[attr-defined]
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color=color, ec=BG, lw=1.5))
        label = tok if len(tok) <= 9 else tok[:8] + "…"
        ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=8.5,
                color="#0d0c0b" if 0.3 < t < 0.7 else INK)
    if len(policy_tokens) > max_tokens:
        ax.text(0, n_rows - 0.02, f"… {len(policy_tokens) - max_tokens} more tokens, truncated for readability",
                color=INK3, fontsize=8.5, va="top")
    ax.set_title("Per-token probability shift after DPO (blue = more likely, red = less likely than the base model)",
                 color=INK, loc="left", fontsize=11.5, pad=14)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tight_run = RES / "dpo-beta1.0.json"
    if tight_run.exists():
        overfitting_trajectory(tight_run, OUT / "overfitting_trajectory.png", beta=1.0)
    metrics = Path("runs/dpo-beta1.0/metrics.jsonl")
    if metrics.exists():
        training_dynamics(metrics, OUT / "training_dynamics.png", title_suffix=" (β=1.0)")
    sweep_cfg = json.loads(Path("configs/dpo-beta-sweep.json").read_text())
    sweep_paths = [RES / f"{r}.json" for r in sweep_cfg["runs"]]
    if all(p.exists() for p in sweep_paths):
        beta_sweep(sweep_paths, sweep_cfg["betas"], OUT / "beta_sweep.png")
    print("[make_figures] wrote figures present in results/ to docs/assets/")
