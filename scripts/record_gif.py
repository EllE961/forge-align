"""Record an animated GIF: base model vs. the strongest DPO run (beta=0.02), generating the
same prompt token by token, with live reward bars that update on the *partial* text each frame.

No browser, no server -- pure matplotlib, rendered frame by frame and encoded with Pillow. The
live "ending" bar is the point: watch it, it stays near zero almost the whole way through for
both models, because the tone lexicon only looks at the last sentence or two and the story
hasn't resolved yet. That's `README.md`'s "ending constraint resolves late" limitation, shown
happening rather than just stated.

    python scripts/record_gif.py --out docs/assets/race.gif
"""

from __future__ import annotations

import argparse
import io
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from align import model_utils, reward  # noqa: E402
from align.prompts import build_prompt  # noqa: E402

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#c0509c", "#3bb273"
BARS = [("keyword", EMBER), ("ending", TEMPER), ("length", PURPLE), ("diversity", GREEN)]

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG,
    "font.family": ["Segoe UI", "DejaVu Sans"],
})


def panel(ax, title: str, text: str, rb: reward.RewardBreakdown, keyword: str, tone: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=PANEL, edgecolor=GRID, lw=1))

    ax.text(0.04, 0.95, title, transform=ax.transAxes, color=INK, fontsize=13, fontweight="bold", va="top")
    ax.text(0.96, 0.95, f"{rb.total * 100:4.0f}%", transform=ax.transAxes, color=EMBER, fontsize=15,
            fontweight="bold", va="top", ha="right")

    wrapped = textwrap.fill(text.strip(), width=46)
    ax.text(0.04, 0.84, wrapped, transform=ax.transAxes, color=INK, fontsize=10.3, va="top", linespacing=1.5)

    for i, (name, color) in enumerate(BARS):
        y = 0.14 - i * 0.038
        val = getattr(rb, name)
        ax.add_patch(plt.Rectangle((0.04, y), 0.62, 0.022, transform=ax.transAxes, facecolor="#1c1a18", lw=0))
        ax.add_patch(plt.Rectangle((0.04, y), 0.62 * val, 0.022, transform=ax.transAxes, facecolor=color, lw=0))
        ax.text(0.68, y + 0.011, name, transform=ax.transAxes, color=INK3, fontsize=8.5, va="center")
        ax.text(0.95, y + 0.011, f"{val * 100:3.0f}%", transform=ax.transAxes, color=INK2, fontsize=8.5,
                va="center", ha="right")


def build_frames(keyword: str, tone: str, seed: int, step: int, hold_last: int) -> list[Image.Image]:
    tok = model_utils.load_tokenizer()
    base = model_utils.load_model(model_utils.DEFAULT_CHECKPOINT)
    aligned = model_utils.load_model("runs/dpo-beta0.02/best.pt")

    prompt = build_prompt(keyword, tone)
    base_text = model_utils.sample_text(base, tok, prompt, seed=seed)
    aligned_text = model_utils.sample_text(aligned, tok, prompt, seed=seed)
    base_ids = tok.encode(base_text)
    aligned_ids = tok.encode(aligned_text)
    n = max(len(base_ids), len(aligned_ids))

    frames = []
    for k in range(2, n + 1, step):
        b_txt = tok.decode(base_ids[: min(k, len(base_ids))])
        a_txt = tok.decode(aligned_ids[: min(k, len(aligned_ids))])
        rb_b = reward.score(b_txt, keyword, tone)
        rb_a = reward.score(a_txt, keyword, tone)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
        fig.suptitle(f'"{prompt.splitlines()[0]}"', color=INK2, fontsize=11, y=0.985)
        panel(ax1, "base (no alignment)", b_txt, rb_b, keyword, tone)
        panel(ax2, "DPO-aligned (β=0.02)", a_txt, rb_a, keyword, tone)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    frames += [frames[-1]] * hold_last
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyword", default="dragon")
    ap.add_argument("--tone", default="happy")
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--step", type=int, default=2, help="tokens revealed per frame")
    ap.add_argument("--hold-last", type=int, default=10, help="extra copies of the final frame")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--out", type=Path, default=Path("docs/assets/race.gif"))
    args = ap.parse_args()

    frames = build_frames(args.keyword, args.tone, args.seed, args.step, args.hold_last)
    # Note for anyone re-verifying this file: PIL.ImageSequence.Iterator reuses one Image object
    # across seeks, so `list(ImageSequence.Iterator(im))` silently gives N references to the same
    # (last) frame. Copy each frame (`.convert("RGB").copy()`) while iterating, or this looks like
    # every frame is identical when the GIF itself is fine -- cost me a debugging detour here.
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True
    )
    print(f"[record_gif] {len(frames)} frames -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
