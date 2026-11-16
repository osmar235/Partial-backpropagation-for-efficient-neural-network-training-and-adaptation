from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"

NAVY = "#244e73"
INK = "#18212a"
ARROW = "#46515a"
BLUE = "#e8f2f8"
PEACH = "#fff0e8"
YELLOW = "#fff5cf"
GREEN = "#e8f5ef"
PINK = "#fbeaea"
GRID = "#d8dde2"


def setup_canvas(width_px: int, height_px: int, dpi: int = 200):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    return fig, ax


def rounded_box(ax, x, y, w, h, text, face, fontsize=11.5, weight="normal", linespacing=1.15):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.06",
        linewidth=1.55,
        edgecolor=NAVY,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=linespacing,
        zorder=3,
    )


def diamond(ax, cx, cy, w, h, text, face, fontsize=11.2):
    points = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(points, closed=True, linewidth=1.55, edgecolor=NAVY, facecolor=face, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=INK, linespacing=1.05, zorder=3)


def orth_arrow(ax, points, label=None, label_xy=None):
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=ARROW, linewidth=1.55, solid_capstyle="round", zorder=1)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.55,
            color=ARROW,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )
    if label and label_xy:
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=13.0,
            color=ARROW,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7),
            zorder=4,
        )


def make_figure_1(path: Path):
    fig, ax = setup_canvas(2562, 1463)
    ax.text(
        7,
        7.63,
        "Loss-gated partial backpropagation with Adam moment continuation",
        ha="center",
        va="center",
        fontsize=20.0,
        fontweight="bold",
        color="#111111",
    )

    rounded_box(ax, 0.35, 6.15, 2.40, 0.95, "Mini-batch forward\nCompute loss  $\\ell_t$", BLUE, fontsize=14.2, weight="bold")
    diamond(ax, 4.25, 6.62, 2.70, 1.18, "Was the previous\nupdate full?", PEACH, fontsize=13.8)
    diamond(ax, 4.25, 4.72, 2.60, 1.18, "Loss gate\n$\\ell_t \\leq m_{t-1}$ ?", YELLOW, fontsize=13.8)
    rounded_box(ax, 10.15, 5.72, 2.85, 1.38, "Full backward\nAll parameter\ngradients", PINK, fontsize=13.3, weight="bold")
    rounded_box(
        ax,
        2.30,
        2.58,
        3.90,
        1.30,
        "Partial backward\nExact fresh suffix gradients\nSet omitted prefix gradient to zero\nContinue Adam prefix moments",
        GREEN,
        fontsize=12.4,
        weight="bold",
        linespacing=1.08,
    )
    rounded_box(
        ax,
        6.55,
        0.95,
        3.25,
        1.15,
        "Ordinary Adam step\nUpdate all parameters",
        BLUE,
        fontsize=14.0,
        weight="bold",
        linespacing=1.08,
    )
    rounded_box(ax, 10.35, 0.95, 3.15, 1.15, "Update loss EMA\nfor the next decision\n$m_t = 0.95m_{t-1} + 0.05\\ell_t$", BLUE, fontsize=11.7, linespacing=1.03)

    # Each decision branch terminates at a backward box. Only the completed
    # backward routes feed the shared optimizer step below.
    orth_arrow(ax, [(2.75, 6.62), (2.90, 6.62)])
    orth_arrow(ax, [(5.60, 6.62), (10.15, 6.62)], "No", (7.85, 6.85))
    orth_arrow(ax, [(4.25, 6.03), (4.25, 5.31)], "Yes", (4.53, 5.66))
    orth_arrow(ax, [(5.55, 4.72), (8.80, 4.72), (8.80, 6.05), (10.15, 6.05)], "No", (5.92, 4.96))
    orth_arrow(ax, [(4.25, 4.13), (4.25, 3.88)], "Yes", (4.53, 4.03))

    # Full and partial backward are distinct completed routes. They converge
    # only at the single ordinary Adam step.
    orth_arrow(ax, [(4.25, 2.58), (4.25, 2.30), (7.35, 2.30), (7.35, 2.10)])
    orth_arrow(ax, [(11.58, 5.72), (11.58, 2.30), (9.00, 2.30), (9.00, 2.10)])
    orth_arrow(ax, [(9.80, 1.53), (10.35, 1.53)])

    ax.text(
        7,
        0.27,
        "Safety rule: every partial update forces the next update to be full, so a prefix loss gradient is omitted for at most one consecutive mini-batch.",
        ha="center",
        va="center",
        fontsize=12.8,
        color=NAVY,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def make_figure_2(path: Path):
    fig, ax = setup_canvas(2615, 839)
    ax.text(7, 7.48, "Development and confirmation sequence", ha="center", va="center", fontsize=20, fontweight="bold")

    boxes = [
        (0.35, "Method\ndevelopment", "Seen seeds\n10, 20, 30", BLUE),
        (3.75, "Frozen short-horizon\nheld-out evaluation", "Seeds 31–39\nAggregate gate failed", PINK),
        (7.25, "Protocol strengthening\ncontroller unchanged", "Seed 40\nAugmentation, cosine schedule,\nand timing qualification", YELLOW),
        (10.80, "Confirmatory\nheld-out test", "Seeds 41–49\nAdam gates passed\nAdan: +2.42 pp\n12.09% faster", GREEN),
    ]
    y, w, h = 2.15, 2.85, 3.55
    for x, heading, detail, face in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.06", linewidth=1.55, edgecolor=NAVY, facecolor=face))
        ax.text(x + w / 2, y + 2.32, heading, ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK, linespacing=1.05)
        ax.text(x + w / 2, y + 0.90, detail, ha="center", va="center", fontsize=12.2, color=INK, linespacing=1.08)
    for x0, x1 in [(3.20, 3.75), (6.60, 7.25), (10.10, 10.80)]:
        orth_arrow(ax, [(x0, y + h / 2), (x1, y + h / 2)])
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def make_figure_3(path: Path):
    fig = plt.figure(figsize=(2582 / 200, 1022 / 200), dpi=200, facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.10, right=0.985, bottom=0.18, top=0.78, wspace=0.55)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    fig.suptitle("Stage 6B confirmatory outcomes (nine held-out seeds)", fontsize=20, fontweight="bold", y=0.95)

    means = [-0.548, 2.424]
    lows = [-0.759, 1.641]
    highs = [-0.337, 3.207]
    y = [1, 0]
    xerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]
    ax1.errorbar(means, y, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=2.1, capsize=5, markersize=7)
    ax1.vlines(-1, 0.64, 1.08, color="#777777", linestyles="--", linewidth=1.35)
    ax1.axvline(0, color="#222222", linewidth=1.35)
    ax1.text(-0.96, 1.045, "Adam pre-specified margin", ha="left", va="center", fontsize=9.5, color="#777777")
    ax1.set_yticks(y, ["Adam", "Tuned Adan"])
    ax1.set_xlim(-1.2, 3.4)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("Final accuracy difference (percentage points)", fontsize=12.5)
    ax1.set_title("Accuracy: LGPB-MC minus baseline", fontsize=15.0, pad=8)

    ratio_means = [0.8976, 0.9208, 0.8791]
    ratio_lows = [0.8743, 0.8714, 0.8613]
    ratio_highs = [0.9215, 0.9730, 0.8972]
    yr = [2, 1, 0]
    ratio_err = [[m - lo for m, lo in zip(ratio_means, ratio_lows)], [hi - m for m, hi in zip(ratio_means, ratio_highs)]]
    ax2.errorbar(ratio_means, yr, xerr=ratio_err, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=2.1, capsize=5, markersize=7)
    ax2.axvline(1, color="#777777", linestyle="--", linewidth=1.35)
    ax2.set_yticks(yr, ["Wall time / Adam", "Time to 80% / Adam", "Wall time / tuned Adan"])
    ax2.set_xlim(0.855, 1.007)
    ax2.set_ylim(-0.1, 2.1)
    ax2.set_xlabel("LGPB-MC / baseline time ratio", fontsize=12.5)
    ax2.set_title("Training efficiency", fontsize=15.0, pad=8)

    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=12.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def replace_docx_media(source_docx: Path, output_docx: Path, replacements: dict[str, Path]):
    temp_docx = output_docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(source_docx, "r") as zin, zipfile.ZipFile(temp_docx, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in replacements:
                data = replacements[item.filename].read_bytes()
            zout.writestr(item, data)
    shutil.move(temp_docx, output_docx)


def main():
    parser = argparse.ArgumentParser(description="Regenerate the three publication figures.")
    parser.add_argument(
        "--source-docx",
        type=Path,
        help="Optional manuscript DOCX whose word/media/image1-3.png files should be replaced.",
    )
    parser.add_argument(
        "--output-docx",
        type=Path,
        help="Output DOCX path. Required when --source-docx is provided.",
    )
    args = parser.parse_args()
    if bool(args.source_docx) != bool(args.output_docx):
        parser.error("--source-docx and --output-docx must be provided together")

    FIGURE_DIR.mkdir(exist_ok=True)
    figure1 = FIGURE_DIR / "Figure_1_LGPB_MC_decision_path.png"
    figure2 = FIGURE_DIR / "Figure_2_development_confirmation_sequence.png"
    figure3 = FIGURE_DIR / "Figure_3_confirmatory_outcomes.png"
    make_figure_1(figure1)
    make_figure_2(figure2)
    make_figure_3(figure3)
    if args.source_docx:
        source_docx = args.source_docx.resolve()
        output_docx = args.output_docx.resolve()
        if source_docx == output_docx:
            parser.error("--output-docx must differ from --source-docx")
        replace_docx_media(
            source_docx,
            output_docx,
            {
                "word/media/image1.png": figure1,
                "word/media/image2.png": figure2,
                "word/media/image3.png": figure3,
            },
        )
        print(output_docx)
    print(figure1)
    print(figure2)
    print(figure3)


if __name__ == "__main__":
    main()
