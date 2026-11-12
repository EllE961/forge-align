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


def reward_hacking(run_json: Path, out: Path) -> None:
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
    l1, = ax1.plot(steps, proxy, color=EMBER, marker="o", label="proxy reward (training signal)")
    l2, = ax2.plot(steps, fluency, color=TEMPER, marker="o", label="true fluency (held out, never trained on)")
    ax1.set_xlabel("DPO step")
    ax1.set_ylabel("proxy reward", color=EMBER)
    ax2.set_ylabel("fluency (mean log p / token)", color=TEMPER)
    ax1.set_title("Reward hacking, measured: proxy keeps climbing, fluency doesn't")
    # annotate the step where fluency's trend turns down while proxy's doesn't
    if len(fluency) > 2:
        drop_idx = int(np.argmin(np.diff(fluency))) + 1
        ax1.annotate("fluency starts falling here", xy=(steps[drop_idx], proxy[drop_idx]),
                     xytext=(10, -18), textcoords="offset points", color=INK2, fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": INK3, "lw": 1})
    ax1.legend(handles=[l1, l2], loc="lower right")
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
        finals.append(max(rows, key=lambda r: r["step"] if r["step"] else 10**9))
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


def token_shift(policy_tokens: list[str], ref_logp: list[float], policy_logp: list[float], out: Path) -> None:
    """A per-token strip, colour-coded by how much more/less likely each token became after DPO."""
    diff = np.array(policy_logp) - np.array(ref_logp)
    vmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
    fig, ax = plt.subplots(figsize=(11, 1.6))
    ax.set_xlim(0, len(policy_tokens))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (tok, d) in enumerate(zip(policy_tokens, diff)):
        t = (d / vmax + 1) / 2  # -> [0, 1]
        color = plt.cm.RdBu_r(t)  # type: ignore[attr-defined]
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color, ec=None))
        ax.text(i + 0.5, 0.5, tok, ha="center", va="center", fontsize=8,
                color="#0d0c0b" if 0.3 < t < 0.7 else INK)
    ax.set_title("Per-token probability shift after DPO (blue = more likely, red = less likely)",
                 color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_run = RES / "dpo-beta0.1.json"
    if main_run.exists():
        reward_hacking(main_run, OUT / "reward_hacking.png")
    metrics = Path("runs/dpo-beta0.1/metrics.jsonl")
    if metrics.exists():
        training_dynamics(metrics, OUT / "training_dynamics.png")
    sweep_cfg = json.loads(Path("configs/dpo-beta-sweep.json").read_text())
    sweep_paths = [RES / f"{r}.json" for r in sweep_cfg["runs"]]
    if all(p.exists() for p in sweep_paths):
        beta_sweep(sweep_paths, sweep_cfg["betas"], OUT / "beta_sweep.png")
    print("[make_figures] wrote figures present in results/ to docs/assets/")
