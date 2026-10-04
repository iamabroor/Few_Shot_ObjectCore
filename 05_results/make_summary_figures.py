#!/usr/bin/env python3
"""Builds the summary figures in 05_results/figures/summary/ and the methodology diagram in 01_methodology/
from the CSVs in 05_results/metrics (no GPU, no dataset needed):  python 05_results/make_summary_figures.py"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = Path(__file__).resolve().parent
FIG = R / "figures" / "summary"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def rows(name):
    return list(csv.DictReader(open(R / "metrics" / name)))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- 1. I-AUROC vs k: paper vs ours ---------------------------------------------------------
s = rows("run1_iauroc_summary.csv")
ks = [1, 2, 4]
paper = [72.5, 74.6, 80.8]; paper_sd = [0.2, 0.6, 1.3]
B = [float(r["i_auroc_mean"]) for r in s if r["phase"].startswith("B")]
Bsd = [float(r["i_auroc_std"]) for r in s if r["phase"].startswith("B")]
A = [float(r["i_auroc_mean"]) for r in s if r["phase"].startswith("A")]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for y, sd, c, lab in [(paper, paper_sd, ORANGE, "ObjectCore – paper"),
                      (B, Bsd, BLUE, "Ours – Phase B (fine-tuned, paper setting)"),
                      (A, None, AQUA, "Ours – Phase A (no fine-tuning)")]:
    ax.errorbar(ks, y, yerr=sd, color=c, lw=2, marker="o", ms=8, capsize=4, mec=SURF, mew=2, label=lab)
    ax.annotate(f"{y[-1]:.1f}", (4, y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK)
ax.axhline(54.7, color=INK2, lw=1, ls=":"); ax.text(1.02, 55.2, "PatchCore k=1 (54.7)", color=INK2, fontsize=8)
ax.set_xticks(ks); ax.set_xlabel("support images k"); ax.set_ylabel("I-AUROC (%)"); ax.set_ylim(50, 85)
ax.set_title("MVTec LOCO – image-level AUROC vs number of support images", color=INK, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
save(fig, "fig_iauroc_vs_k")

# ---- 2. k=4 per category: full model vs branch-only ------------------------------------------
c = [r for r in rows("run1_phaseB_k4_per_category.csv")]
names = [r["category"].replace("_", "\n") for r in c]
series = [("iauroc_all_mean", BLUE, "ObjectCore (A_S + A_L)"),
          ("structural_only_branch", ORANGE, "only structural bank M_S"),
          ("logical_only_branch", AQUA, "only logical bank M_L")]
fig, ax = plt.subplots(figsize=(9, 4))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
w = 0.26
for i, (key, col, lab) in enumerate(series):
    xs = [j + (i - 1) * w for j in range(len(c))]
    vals = [float(r[key]) for r in c]
    ax.bar(xs, [v - 40 for v in vals], bottom=40, width=w - 0.03, color=col, label=lab)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.6, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.axhline(80.8, color=INK2, lw=1, ls="--"); ax.text(len(c) - 0.5, 81.5, "paper average 80.8", ha="right", color=INK2, fontsize=8)
ax.set_xticks(range(len(c))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(40, 90); ax.set_ylabel("I-AUROC (%), k = 4, mean of 3 draws")
ax.set_title("Run 1, Phase B – the structural branch alone beats the full model in every category",
             color=INK, loc="left")
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)
save(fig, "fig_k4_branches_per_category")

# ---- 3. detector collapse diagnostic -----------------------------------------------------------
d = rows("diag_finetune_summary.csv")
labels = ["pretrained\n(pseudo-labels)", "V1: paper setting\n(HF loss, all tokens)", "V2: our fix\n(phrase-token loss)"]
sup = [int(r["objects_on_support_after_ft"]) for r in d]
tst = [None, 1, 6]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.bar([x - 0.18 for x in range(3)], sup, width=0.34, color=BLUE, label="support image (266.png)")
ax.bar([x + 0.18 for x in range(1, 3)], tst[1:], width=0.34, color=ORANGE, label="each of 3 test images")
for x, v in enumerate(sup):
    ax.text(x - 0.18, v + 0.15, str(v), ha="center", color=INK)
for x, v in zip(range(1, 3), tst[1:]):
    ax.text(x + 0.18, v + 0.15, str(v), ha="center", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0, 10)
ax.set_ylabel("objects detected (threshold 0.25)")
ax.set_title("breakfast_box – objects kept after 100 epochs of fine-tuning", color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
save(fig, "fig_detector_collapse_diagnostic")

# ---- 4. methodology diagram ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 6.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=col, lw=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=col, ec=col, lw=2))
    ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + (h - 0.42) / 2, body, ha="center", va="center", color=INK, fontsize=8.2, linespacing=1.35)


def arrow(x0, y0, x1, y1, col=INK2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))


ax.text(0.1, 5.95, "ObjectCore – few-shot logical anomaly detection with object representations (reimplementation)",
        fontsize=13, weight="bold", color=INK)
ax.text(0.1, 5.6, "Top row: built once from the k normal support images.   Bottom row: every test image.",
        fontsize=9.5, color=INK2)
G = "#4a3aa7"
box(0.1, 3.2, 2.2, 2.1, "1  Support set", "k = 1, 2, 4 normal\nimages (train/good)\n3 random draws\nseed: 1000·k + s", INK2)
box(2.7, 3.2, 2.5, 2.1, "2  GroundingDINO", 'Swin-B, 800×800\nprompt "part"\nbox / text thr 0.25\n→ pseudo-boxes', BLUE)
box(5.6, 3.2, 2.5, 2.1, "3  Fine-tune detector", "100 epochs, AdamW\nlr 1e-5, batch 1\nL_cls only (no L_bbox)\nfix: loss on \"part\" token", ORANGE)
box(8.5, 3.2, 2.5, 2.1, "4  SAM + features", "SAM ViT-B mask per box\nSwin layers 1–3, 5×5 mean\no = mean feature in mask\n+ mask size m", AQUA)
box(11.4, 4.3, 3.5, 1.0, "5a  Structural bank M_S", "all support patch features (no coreset)", G)
box(11.4, 3.2, 3.5, 1.0, "5b  Logical bank M_L", "key: mean image feature → value: object set", G)
for x0, x1 in [(2.3, 2.7), (5.2, 5.6), (8.1, 8.5)]:
    arrow(x0, 4.25, x1, 4.25)
arrow(11.0, 4.6, 11.4, 4.8); arrow(11.0, 3.9, 11.4, 3.7)

box(0.1, 0.2, 2.2, 2.3, "Test image", "any test image\n(good / logical /\nstructural)", INK2)
box(2.7, 0.2, 2.5, 2.3, "Describe", "fine-tuned detector\n→ SAM masks\n→ patch features\n→ objects (o, m)", BLUE)
box(5.6, 1.4, 2.9, 1.1, "A_S  structural map", "1 − max cos-sim to M_S per patch", ORANGE)
box(5.6, 0.2, 2.9, 1.1, "A_L  logical map", "nearest support image, Hungarian\ncost = cos-dist + 0.1·|Δsize|", AQUA)
box(8.9, 0.2, 2.9, 2.3, "Anomaly map", "A = Gauss_σ=4 (A_S + A_L)\nunmatched objects add\ntheir whole mask\nscore = mean top 1 %", G)
box(12.2, 0.2, 2.7, 2.3, "Decision", "score > threshold\n→ ANOMALY\nelse → NORMAL\n(+ pixel heat map)", "#e34948")
arrow(2.3, 1.35, 2.7, 1.35); arrow(5.2, 1.6, 5.6, 1.95); arrow(5.2, 1.1, 5.6, 0.75)
arrow(8.5, 1.95, 8.9, 1.6); arrow(8.5, 0.75, 8.9, 1.1); arrow(11.8, 1.35, 12.2, 1.35)
arrow(12.2, 3.2, 7.05, 2.55, G)
ax.text(3.0, 2.78, "memory banks from the top row:  M_S → A_S,  M_L → A_L", color=G, fontsize=8.5, style="italic")
out = R.parent / "01_methodology" / "methodology_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
print("figures written to", FIG, "and", out)
