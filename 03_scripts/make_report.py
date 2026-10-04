#!/usr/bin/env python3
"""
ObjectCore replication report: comparison CSVs, graphs and paper-style figures for one run folder.

    python make_report.py                                   # newest run (~/objectcore/runs/latest)
    python make_report.py ~/objectcore/runs/2026-10-01_2030_all

Output: <run>/report/
    tables/   table1_comparison.csv          paper Table 1 (MVTec LOCO) vs this reimplementation, k = 1, 2, 4
              table1_comparison_long.csv     same, one row per (k, metric) with gaps
              per_category.csv               every category x k x phase, all metrics, mean +- std over draws
              ablation_comparison.csv        paper Table 3 rows we can reproduce (Only MS / Only ML / No fine-tuning)
              per_draw.csv                   every single run, incl. the support images used
              context_fullshot.csv           vs own SALAD / LogiCo replications (full-shot, different metric)
    graphs/   g1_kshot_curves.png            4 metrics vs k: paper vs ours (+ PatchCore / PromptAD from the paper)
              g2_per_category_k4.png         per-category bars, k = 4
              g3_ablation.png                paper Table 3 vs ours
              g4_roc_curves.png              ROC per category (good vs logical / good vs structural)
              g5_score_distributions.png     anomaly-score histograms per category
              g6_objects_per_image.png       objects found per image, before vs after fine-tuning
    figures/  fig2_detections.png            paper Fig. 2 style: detections before / after fine-tuning + SAM masks
              fig3_qualitative_best.png      paper Fig. 3 style: image / structural / logical / ObjectCore / GT
              fig3_qualitative_random.png    same, randomly chosen examples (no cherry-picking)
              per_category/<cat>_examples.png  more examples per category
    report.html                              everything on one page
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.metrics import roc_curve

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
CAT_SHORT = {"breakfast_box": "Breakfast box", "juice_bottle": "Juice bottle", "pushpins": "Pushpins",
             "screw_bag": "Screw bag", "splicing_connectors": "Splicing conn."}
METRICS = ["I-AUROC", "I-F1-max", "P-AUROC", "P-F1-max"]
PHASES = {"B_finetune": "ObjectCore (ours, reimpl.)", "A_zeroshot": "ObjectCore w/o fine-tuning (ours)"}

# ---- numbers from the paper (WACV 2026, Table 1 / Table 3 / supplementary) --------------------
PAPER_T1 = {  # MVTec LOCO: method -> k -> metric -> (mean, std)
    "ObjectCore (paper)": {1: {"I-AUROC": (72.5, 0.2), "I-F1-max": (79.8, 1.0), "P-AUROC": (76.3, 0.3), "P-F1-max": (28.8, 1.7)},
                           2: {"I-AUROC": (74.6, 0.6), "I-F1-max": (80.5, 0.6), "P-AUROC": (81.4, 0.5), "P-F1-max": (29.3, 0.8)},
                           4: {"I-AUROC": (80.8, 1.3), "I-F1-max": (82.8, 0.4), "P-AUROC": (84.4, 0.6), "P-F1-max": (31.9, 1.2)}},
    "PatchCore (paper)": {1: {"I-AUROC": (54.7, None), "I-F1-max": (77.4, None), "P-AUROC": (75.5, None), "P-F1-max": (12.5, None)},
                          2: {"I-AUROC": (58.3, None), "I-F1-max": (77.5, None), "P-AUROC": (80.8, None), "P-F1-max": (14.6, None)},
                          4: {"I-AUROC": (63.5, None), "I-F1-max": (77.7, None), "P-AUROC": (82.8, None), "P-F1-max": (15.3, None)}},
    "PromptAD (paper)": {1: {"I-AUROC": (61.9, None), "I-F1-max": (77.5, None), "P-AUROC": (64.3, None), "P-F1-max": (14.4, None)},
                         2: {"I-AUROC": (63.8, None), "I-F1-max": (78.0, None), "P-AUROC": (65.9, None), "P-F1-max": (15.2, None)},
                         4: {"I-AUROC": (70.9, None), "I-F1-max": (78.3, None), "P-AUROC": (72.8, None), "P-F1-max": (15.8, None)}},
}
PAPER_ABL = {  # Table 3, k = 4, differences to the full model
    "Full model": {"I-AUROC": 0.0, "I-F1-max": 0.0, "P-AUROC": 0.0, "P-F1-max": 0.0},
    "Only M_S (structural branch only)": {"I-AUROC": -12.1, "I-F1-max": -5.2, "P-AUROC": -4.5, "P-F1-max": -2.2},
    "Only M_L (logical branch only)": {"I-AUROC": -10.2, "I-F1-max": -4.3, "P-AUROC": -2.4, "P-F1-max": -1.3},
    "No fine-tuning": {"I-AUROC": -11.8, "I-F1-max": -2.7, "P-AUROC": -4.2, "P-F1-max": -1.9},
}
PAPER_PER_CAT_K4 = {"screw_bag": {"I-AUROC": 72.3, "I-F1-max": 79.3}}  # supplementary Table 4
# own full-shot replications (metric = mean of logical and structural I-AUROC) - context only
FULLSHOT = {"SALAD replication (full-shot, seed 42)": {"breakfast_box": 86.09, "juice_bottle": 99.30, "pushpins": 98.89, "screw_bag": 88.54, "splicing_connectors": 96.18, "AVERAGE": 93.80},
            "LogiCo replication (full-shot, epoch 200)": {"breakfast_box": 98.86, "juice_bottle": 97.69, "pushpins": 93.35, "screw_bag": 85.29, "splicing_connectors": 94.20, "AVERAGE": 93.88}}

COL = {"paper": "#6b6b6b", "B_finetune": "#1f5fbf", "A_zeroshot": "#e08a1e", "PatchCore": "#9a9a9a", "PromptAD": "#b9b9b9",
       "logical": "#c0392b", "structural": "#2e86ab", "good": "#4caf50"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


# ----------------------------------------------------------------------------------------- load
def load_runs(results_root):
    runs = []
    for f in sorted(results_root.glob("*/*/k*_seed*/results.json")):
        r = json.loads(f.read_text())
        r["_dir"] = f.parent
        runs.append(r)
    return runs


def ms(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), len(v)


def fmt(m, s=None):
    if m is None or np.isnan(m):
        return ""
    return f"{m:.1f}" if s is None or np.isnan(s) else f"{m:.1f} ± {s:.1f}"


def metric_of(r, metric, branch=None):
    src = r.get("metrics_branch_only", {}).get(branch, {}) if branch else r.get("metrics", {})
    if metric == "I-AUROC" and not src and not branch:
        return r["auroc"]["all"]
    return src.get(metric, float("nan"))


def averaged(runs, phase, k, metric, branch=None):
    """Paper protocol: average the 5 categories per draw, then mean +- std over draws."""
    by_seed = defaultdict(dict)
    for r in runs:
        if r["phase"] == phase and r["k"] == k:
            by_seed[r["seed"]][r["category"]] = metric_of(r, metric, branch)
    per_draw = [np.mean([d[c] for c in CATS]) for d in by_seed.values() if all(c in d for c in CATS)]
    return ms(per_draw)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------------------- tables
def make_tables(runs, out):
    ks = sorted({r["k"] for r in runs})
    phases = [p for p in PHASES if any(r["phase"] == p for r in runs)]

    # Table 1, wide
    rows = []
    for k in [1, 2, 4]:
        for meth, d in PAPER_T1.items():
            rows.append([k, meth, "paper"] + sum([[d[k][m][0], d[k][m][1] if d[k][m][1] is not None else ""] for m in METRICS], []) + ["", ""])
        for ph in phases:
            vals, n = [], 0
            for m in METRICS:
                mean, sd, n_ = averaged(runs, ph, k, m)
                n = max(n, n_)
                vals += [round(mean, 2) if not np.isnan(mean) else "", round(sd, 2) if not np.isnan(sd) else ""]
            gap = averaged(runs, ph, k, "I-AUROC")[0] - PAPER_T1["ObjectCore (paper)"][k]["I-AUROC"][0]
            rows.append([k, PHASES[ph], "this replication"] + vals + [n, round(gap, 2) if not np.isnan(gap) else ""])
    hdr = ["k", "method", "source"] + sum([[m, m + "_std"] for m in METRICS], []) + ["n_draws", "I-AUROC_gap_to_paper"]
    write_csv(out / "table1_comparison.csv", hdr, rows)

    # Table 1, long
    rows = []
    for k in [1, 2, 4]:
        for m in METRICS:
            pm, ps = PAPER_T1["ObjectCore (paper)"][k][m]
            row = [k, m, pm, ps]
            for ph in ["B_finetune", "A_zeroshot"]:
                mean, sd, n = averaged(runs, ph, k, m)
                row += ["" if np.isnan(mean) else round(mean, 2), "" if np.isnan(sd) else round(sd, 2),
                        "" if np.isnan(mean) else round(mean - pm, 2), n]
            row += [PAPER_T1["PatchCore (paper)"][k][m][0], PAPER_T1["PromptAD (paper)"][k][m][0]]
            rows.append(row)
    write_csv(out / "table1_comparison_long.csv",
              ["k", "metric", "paper_ObjectCore", "paper_std", "ours_finetune", "ours_finetune_std", "gap_finetune_vs_paper",
               "n_draws_finetune", "ours_no_finetune", "ours_no_finetune_std", "gap_no_finetune_vs_paper", "n_draws_no_finetune",
               "paper_PatchCore", "paper_PromptAD"], rows)

    # per category
    rows = []
    for ph in phases:
        for k in ks:
            for c in CATS:
                rs = [r for r in runs if r["phase"] == ph and r["k"] == k and r["category"] == c]
                if not rs:
                    continue
                row = [PHASES[ph], k, c, len(rs)]
                for m in METRICS:
                    row.append(fmt(*ms([metric_of(r, m) for r in rs])[:2]))
                for sub in ["logical", "structural", "mean_LS"]:
                    row.append(fmt(*ms([r["auroc"][sub] for r in rs])[:2]))
                for br in ["structural_only", "logical_only"]:
                    row.append(fmt(*ms([metric_of(r, "I-AUROC", br) for r in rs])[:2]))
                row.append(fmt(*ms([r["mean_objects_per_test"] for r in rs])[:2]))
                p = PAPER_PER_CAT_K4.get(c, {}) if k == 4 and ph == "B_finetune" else {}
                row += [p.get("I-AUROC", ""), p.get("I-F1-max", "")]
                rows.append(row)
    write_csv(out / "per_category.csv",
              ["method", "k", "category", "n_draws"] + METRICS +
              ["I-AUROC_logical", "I-AUROC_structural", "I-AUROC_mean_LS", "I-AUROC_structural_branch_only",
               "I-AUROC_logical_branch_only", "objects_per_test_image", "paper_I-AUROC", "paper_I-F1-max"], rows)

    # ablation (k = 4)
    rows = []
    full = {m: averaged(runs, "B_finetune", 4, m)[0] for m in METRICS}
    ours = {"Full model": full,
            "Only M_S (structural branch only)": {m: averaged(runs, "B_finetune", 4, m, "structural_only")[0] for m in METRICS},
            "Only M_L (logical branch only)": {m: averaged(runs, "B_finetune", 4, m, "logical_only")[0] for m in METRICS},
            "No fine-tuning": {m: averaged(runs, "A_zeroshot", 4, m)[0] for m in METRICS}}
    base = {m: PAPER_T1["ObjectCore (paper)"][4][m][0] for m in METRICS}
    for cond, d in PAPER_ABL.items():
        row = [cond]
        for m in METRICS:
            o = ours[cond][m]
            row += [d[m], round(base[m] + d[m], 1), "" if np.isnan(o) else round(o, 2),
                    "" if np.isnan(o) or np.isnan(full[m]) else round(o - full[m], 2)]
        rows.append(row)
    write_csv(out / "ablation_comparison.csv", ["condition (k=4)"] + sum(
        [[f"paper_delta_{m}", f"paper_{m}", f"ours_{m}", f"ours_delta_{m}"] for m in METRICS], []), rows)

    # every draw
    rows = []
    for r in sorted(runs, key=lambda r: (r["phase"], r["k"], r["category"], r["seed"])):
        rows.append([r["phase"], r["k"], r["seed"], r["category"]] + [round(metric_of(r, m), 2) for m in METRICS] +
                    [round(r["auroc"][s], 2) for s in ["logical", "structural", "mean_LS"]] +
                    [round(metric_of(r, "I-AUROC", "structural_only"), 2), round(metric_of(r, "I-AUROC", "logical_only"), 2),
                     " ".join(Path(p).name for p in r["support_images"]), r["objects_per_support"],
                     round(r["mean_objects_per_test"], 2), r["n_test"], r["runtime_s"]])
    write_csv(out / "per_draw.csv", ["phase", "k", "seed", "category"] + METRICS +
              ["I-AUROC_logical", "I-AUROC_structural", "I-AUROC_mean_LS", "I-AUROC_structural_branch_only",
               "I-AUROC_logical_branch_only", "support_images", "objects_per_support", "objects_per_test", "n_test", "runtime_s"], rows)

    # context: full-shot replications (mean of logical + structural I-AUROC)
    rows = []
    for name, d in FULLSHOT.items():
        rows.append([name, "full train set"] + [d[c] for c in CATS] + [d["AVERAGE"]])
    for ph in phases:
        for k in ks:
            vals = []
            for c in CATS:
                rs = [r["auroc"]["mean_LS"] for r in runs if r["phase"] == ph and r["k"] == k and r["category"] == c]
                vals.append(round(np.mean(rs), 2) if rs else "")
            avg = round(np.mean(vals), 2) if all(v != "" for v in vals) else ""
            rows.append([PHASES[ph], f"{k} images"] + vals + [avg])
    write_csv(out / "context_fullshot.csv", ["method", "training data"] + CATS + ["AVERAGE (metric: mean of logical & structural I-AUROC)"], rows)


# --------------------------------------------------------------------------------------- graphs
def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_graphs(runs, out):
    phases = [p for p in PHASES if any(r["phase"] == p for r in runs)]
    # g1: metric vs k
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, m in zip(axes, METRICS):
        ks = [1, 2, 4]
        for meth, col, ls in [("PatchCore (paper)", COL["PatchCore"], ":"), ("PromptAD (paper)", COL["PromptAD"], ":")]:
            ax.plot(ks, [PAPER_T1[meth][k][m][0] for k in ks], ls, marker="s", ms=4, color=col, label=meth)
        pm = [PAPER_T1["ObjectCore (paper)"][k][m] for k in ks]
        ax.errorbar(ks, [p[0] for p in pm], yerr=[p[1] for p in pm], fmt="--o", color=COL["paper"], capsize=3, label="ObjectCore (paper)")
        for ph in phases:
            v = [averaged(runs, ph, k, m) for k in ks]
            kk = [k for k, x in zip(ks, v) if not np.isnan(x[0])]
            if kk:
                ax.errorbar(kk, [x[0] for x in v if not np.isnan(x[0])], yerr=[x[1] for x in v if not np.isnan(x[0])],
                            fmt="-o", color=COL[ph], capsize=3, lw=2, label=PHASES[ph])
        ax.set_xticks(ks); ax.set_xlabel("k (support images)"); ax.set_title(m); ax.grid(alpha=.3)
    axes[0].set_ylabel("%  (MVTec LOCO, 5-category mean)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("MVTec LOCO few-shot results: paper vs this replication (mean ± std over random support draws)")
    save(fig, out / "g1_kshot_curves.png")

    # g2: per-category, k = 4
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    x = np.arange(len(CATS))
    ax = axes[0]
    w = 0.38
    for i, ph in enumerate(phases):
        v = [ms([metric_of(r, "I-AUROC") for r in runs if r["phase"] == ph and r["k"] == 4 and r["category"] == c]) for c in CATS]
        ax.bar(x + (i - 0.5) * w, [a[0] for a in v], w, yerr=[a[1] for a in v], capsize=3, color=COL[ph], label=PHASES[ph])
        for xi, a in zip(x + (i - 0.5) * w, v):
            if not np.isnan(a[0]):
                ax.text(xi, a[0] + 1, f"{a[0]:.1f}", ha="center", fontsize=8)
    ax.axhline(80.8, ls="--", color=COL["paper"], label="paper mean, 5 categories (80.8)")
    if "screw_bag" in PAPER_PER_CAT_K4:
        xi = CATS.index("screw_bag")
        ax.plot([xi + 0.3], [72.3], marker="*", ms=14, color="black", ls="", label="paper, screw bag (72.3, supp.)")
    ax.set_xticks(x); ax.set_xticklabels([CAT_SHORT[c] for c in CATS]); ax.set_ylim(40, 105)
    ax.set_ylabel("I-AUROC (%)"); ax.set_title("I-AUROC per category, k = 4"); ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    ph = "B_finetune" if "B_finetune" in phases else phases[0]
    for i, (sub, col) in enumerate([("logical", COL["logical"]), ("structural", COL["structural"])]):
        v = [ms([r["auroc"][sub] for r in runs if r["phase"] == ph and r["k"] == 4 and r["category"] == c]) for c in CATS]
        ax.bar(x + (i - 0.5) * w, [a[0] for a in v], w, yerr=[a[1] for a in v], capsize=3, color=col, label=f"{sub} anomalies")
    ax.set_xticks(x); ax.set_xticklabels([CAT_SHORT[c] for c in CATS]); ax.set_ylim(40, 105)
    ax.set_title(f"{PHASES[ph]}, k = 4: logical vs structural"); ax.legend(fontsize=8, loc="lower left")
    save(fig, out / "g2_per_category_k4.png")

    # g3: ablation
    fig, ax = plt.subplots(figsize=(9, 3.8))
    conds = list(PAPER_ABL)
    full = averaged(runs, "B_finetune", 4, "I-AUROC")[0]
    ours = [full, averaged(runs, "B_finetune", 4, "I-AUROC", "structural_only")[0],
            averaged(runs, "B_finetune", 4, "I-AUROC", "logical_only")[0], averaged(runs, "A_zeroshot", 4, "I-AUROC")[0]]
    paper = [80.8 + PAPER_ABL[c]["I-AUROC"] for c in conds]
    y = np.arange(len(conds))
    ax.barh(y + 0.2, paper, 0.4, color=COL["paper"], label="paper (Table 3)")
    ax.barh(y - 0.2, ours, 0.4, color=COL["B_finetune"], label="this replication")
    for yi, v in zip(y + 0.2, paper):
        ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8)
    for yi, v in zip(y - 0.2, ours):
        if not np.isnan(v):
            ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(conds); ax.invert_yaxis(); ax.set_xlim(40, 100)
    ax.set_xlabel("I-AUROC (%), MVTec LOCO, k = 4"); ax.set_title("Ablation: paper vs this replication"); ax.legend(fontsize=8)
    save(fig, out / "g3_ablation.png")

    # g4 + g5: ROC and score histograms (k = 4, first draw)
    ph = "B_finetune" if "B_finetune" in phases else phases[0]
    sel = {c: next((r for r in sorted(runs, key=lambda r: r["seed"]) if r["phase"] == ph and r["k"] == 4 and r["category"] == c), None) for c in CATS}
    fig4, ax4 = plt.subplots(1, 5, figsize=(18, 3.6))
    fig5, ax5 = plt.subplots(1, 5, figsize=(18, 3.2))
    for i, c in enumerate(CATS):
        r = sel[c]
        if r is None:
            ax4[i].set_axis_off(); ax5[i].set_axis_off(); continue
        rows = [json.loads(l) for l in open(r["_dir"] / "scores.jsonl")]
        s = np.array([x["score"] for x in rows]); kind = np.array([x["kind"] for x in rows])
        for sub in ["logical", "structural"]:
            m = (kind == "good") | (kind == sub)
            if (kind == sub).any():
                fpr, tpr, _ = roc_curve(kind[m] == sub, s[m])
                ax4[i].plot(fpr, tpr, color=COL[sub], label=f"{sub} ({r['auroc'][sub]:.1f})")
        ax4[i].plot([0, 1], [0, 1], ":", color="grey"); ax4[i].set_title(CAT_SHORT[c]); ax4[i].legend(fontsize=7, loc="lower right")
        ax4[i].set_xlabel("FPR"); ax4[0].set_ylabel("TPR")
        bins = np.linspace(s.min(), s.max(), 30)
        for sub in ["good", "logical", "structural"]:
            ax5[i].hist(s[kind == sub], bins=bins, alpha=.55, color=COL[sub], label=sub)
        ax5[i].set_title(CAT_SHORT[c]); ax5[i].set_xlabel("anomaly score")
    ax5[0].legend(fontsize=8)
    fig4.suptitle(f"ROC curves, {PHASES[ph]}, k = 4, first support draw")
    fig5.suptitle(f"Anomaly-score distributions, {PHASES[ph]}, k = 4, first support draw")
    save(fig4, out / "g4_roc_curves.png"); save(fig5, out / "g5_score_distributions.png")

    # g6: objects per test image before / after fine-tuning (related to paper Table 2 / Fig. 2)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, ph in enumerate(phases):
        v = [ms([r["mean_objects_per_test"] for r in runs if r["phase"] == ph and r["k"] == 4 and r["category"] == c]) for c in CATS]
        ax.bar(x + (i - 0.5) * 0.38, [a[0] for a in v], 0.38, yerr=[a[1] for a in v], capsize=3, color=COL[ph],
               label="before fine-tuning" if ph == "A_zeroshot" else "after fine-tuning")
    ax.set_xticks(x); ax.set_xticklabels([CAT_SHORT[c] for c in CATS]); ax.set_ylabel("objects per test image")
    ax.set_title("Detected objects per test image (prompt 'part', threshold 0.25), k = 4"); ax.legend(fontsize=8)
    save(fig, out / "g6_objects_per_image.png")


# -------------------------------------------------------------------------------------- figures
def load_maps(run):
    d = run["_dir"] / "maps"
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        out.append({k: z[k] for k in z.files})
    return out


def gt_full(data_root, rel, kind, size):
    W, H = size
    if kind == "good":
        return np.zeros((H, W), bool)
    p = Path(data_root) / rel
    sub = "logical_anomalies" if kind == "logical" else "structural_anomalies"
    d = p.parents[2] / "ground_truth" / sub / p.stem
    m = np.zeros((H, W), bool)
    for f in sorted(d.glob("*.png")) if d.exists() else []:
        m |= np.array(Image.open(f).convert("L").resize((W, H), Image.NEAREST)) > 0
    return m


def to_img_size(a, size):
    return np.array(Image.fromarray(a.astype(np.float32)).resize(size, Image.BILINEAR))


def overlay(ax, img, amap, vmin, vmax):
    ax.imshow(img)
    ax.imshow(np.clip((amap - vmin) / max(vmax - vmin, 1e-8), 0, 1), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_axis_off()


def pick(maps, kind, n, mode, rng):
    idx = [i for i, m in enumerate(maps) if str(m["kind"]) == kind]
    if not idx:
        return []
    if mode == "best":
        idx = sorted(idx, key=lambda i: -float(np.sort(maps[i]["A"].ravel())[-max(1, maps[i]["A"].size // 100):].mean()))
    else:
        rng.shuffle(idx)
    return idx[:n]


def qualitative(runs, data_root, out, mode, per_cat=(1, 1, 0)):
    """Paper Fig. 3 layout: rows = image / structural branch / logical branch / ObjectCore / GT."""
    ph = "B_finetune" if any(r["phase"] == "B_finetune" and r.get("maps_saved") for r in runs) else "A_zeroshot"
    rng = np.random.default_rng(0)
    cols = []
    norms = {}
    for c in CATS:
        r = next((r for r in runs if r["phase"] == ph and r["category"] == c and r.get("maps_saved")), None)
        if r is None:
            continue
        maps = load_maps(r)
        if not maps:
            continue
        allA = np.concatenate([m["A"].ravel() for m in maps]).astype(np.float32)
        norms[c] = (np.percentile(allA, 1), np.percentile(allA, 99.9))
        for kind, n in zip(["logical", "structural", "good"], per_cat):
            for i in pick(maps, kind, n, mode, rng):
                cols.append((c, maps[i], r))
    if not cols:
        return None
    rows = ["Image", "Structural branch $A_S$\n(PatchCore-like)", "Logical branch $A_L$", "ObjectCore $A$", "Ground truth"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.1 * len(cols), 2.1 * len(rows)))
    axes = np.array(axes).reshape(len(rows), len(cols))
    for j, (c, m, r) in enumerate(cols):
        img = Image.open(Path(data_root) / str(m["image"])).convert("RGB")
        size = img.size
        vmin, vmax = norms[c]
        axes[0, j].imshow(img); axes[0, j].set_axis_off()
        axes[0, j].set_title(f"{CAT_SHORT[c]}\n{m['kind']}", fontsize=8)
        for row, key in [(1, "AS"), (2, "AL"), (3, "A")]:
            a = to_img_size(m[key].astype(np.float32), size)
            if key == "A":
                overlay(axes[row, j], img, a, vmin, vmax)
            else:
                lo, hi = np.percentile(m[key].astype(np.float32), 1), np.percentile(m[key].astype(np.float32), 99.9)
                overlay(axes[row, j], img, a, lo, max(hi, vmax if key == "A" else hi))
        g = gt_full(data_root, str(m["image"]), str(m["kind"]), size)
        axes[4, j].imshow(img); axes[4, j].imshow(np.ma.masked_where(~g, g), cmap="autumn", alpha=0.6); axes[4, j].set_axis_off()
    for i, name in enumerate(rows):
        axes[i, 0].text(-0.08, 0.5, name, transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=9)
    sel = "highest-scoring anomalies of each type" if mode == "best" else "randomly chosen anomalies (seed 0)"
    fig.suptitle(f"MVTec LOCO qualitative results ({PHASES[ph]}, k = 4, first draw) – {sel}", y=1.0)
    fig.tight_layout()
    path = out / f"fig3_qualitative_{mode}.png"
    save(fig, path)
    return path


def per_category_examples(runs, data_root, out):
    (out / "per_category").mkdir(exist_ok=True)
    ph = "B_finetune" if any(r["phase"] == "B_finetune" and r.get("maps_saved") for r in runs) else "A_zeroshot"
    rng = np.random.default_rng(1)
    for c in CATS:
        r = next((r for r in runs if r["phase"] == ph and r["category"] == c and r.get("maps_saved")), None)
        if r is None:
            continue
        maps = load_maps(r)
        if not maps:
            continue
        allA = np.concatenate([m["A"].ravel() for m in maps]).astype(np.float32)
        vmin, vmax = np.percentile(allA, 1), np.percentile(allA, 99.9)
        sel = pick(maps, "good", 2, "random", rng) + pick(maps, "logical", 4, "random", rng) + pick(maps, "structural", 4, "random", rng)
        fig, axes = plt.subplots(3, len(sel), figsize=(2.1 * len(sel), 6.6))
        axes = np.array(axes).reshape(3, len(sel))
        scores = {json.loads(l)["image"]: json.loads(l)["score"] for l in open(r["_dir"] / "scores.jsonl")}
        for j, i in enumerate(sel):
            m = maps[i]
            img = Image.open(Path(data_root) / str(m["image"])).convert("RGB")
            axes[0, j].imshow(img); axes[0, j].set_axis_off()
            axes[0, j].set_title(f"{m['kind']}\nscore {scores.get(str(m['image']), float('nan')):.3f}", fontsize=8)
            overlay(axes[1, j], img, to_img_size(m["A"].astype(np.float32), img.size), vmin, vmax)
            g = gt_full(data_root, str(m["image"]), str(m["kind"]), img.size)
            axes[2, j].imshow(img); axes[2, j].imshow(np.ma.masked_where(~g, g), cmap="autumn", alpha=.6); axes[2, j].set_axis_off()
        for i, name in enumerate(["Image", "ObjectCore $A$", "Ground truth"]):
            axes[i, 0].text(-0.08, 0.5, name, transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=9)
        fig.suptitle(f"{CAT_SHORT[c]} – random examples ({PHASES[ph]}, k = 4, first draw)")
        fig.tight_layout()
        save(fig, out / "per_category" / f"{c}_examples.png")


def detections(runs, data_root, out):
    """Paper Fig. 2 layout: detections before fine-tuning / after fine-tuning (+ SAM masks after fine-tuning)."""
    def first_draw(ph, c):
        rs = [r for r in runs if r["phase"] == ph and r["category"] == c and r["k"] == 4]
        return min(rs, key=lambda r: r["seed"]) if rs else None
    cols = []
    for c in CATS:
        a, b = first_draw("A_zeroshot", c), first_draw("B_finetune", c)
        if a is None:
            continue
        rows_a = {json.loads(l)["image"]: json.loads(l) for l in open(a["_dir"] / "scores.jsonl")}
        rows_b = {json.loads(l)["image"]: json.loads(l) for l in open(b["_dir"] / "scores.jsonl")} if b else {}
        img_rel = next((k for k, v in rows_a.items() if v["kind"] == "good"), next(iter(rows_a)))
        objmap = None
        if b is not None and (b["_dir"] / "maps").exists():
            for f in sorted((b["_dir"] / "maps").glob("*.npz")):
                z = np.load(f)
                if str(z["image"]) == img_rel:
                    objmap = z["objmap"]; break
        cols.append((c, img_rel, rows_a[img_rel], rows_b.get(img_rel), objmap))
    if not cols:
        return
    nrows = 3 if any(col[4] is not None for col in cols) else 2
    fig, axes = plt.subplots(nrows, len(cols), figsize=(3.2 * len(cols), 3.0 * nrows))
    axes = np.array(axes).reshape(nrows, len(cols))
    for j, (c, rel, ra, rb, objmap) in enumerate(cols):
        img = Image.open(Path(data_root) / rel).convert("RGB")
        W, H = img.size
        for i, rr in enumerate([ra, rb]):
            ax = axes[i, j]; ax.imshow(img); ax.set_axis_off()
            if rr is None:
                ax.text(.5, .5, "no fine-tuned run", transform=ax.transAxes, ha="center", color="w"); continue
            for b in rr["boxes_norm"]:
                ax.add_patch(mpatches.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H,
                                                fill=False, lw=1.2, color="#00e5ff"))
            ax.set_title(f"{CAT_SHORT[c]}: {len(rr['boxes_norm'])} objects" if i == 0 else f"{len(rr['boxes_norm'])} objects", fontsize=8)
        if nrows == 3:
            ax = axes[2, j]; ax.imshow(img); ax.set_axis_off()
            if objmap is not None and objmap.max() > 0:
                om = np.array(Image.fromarray(objmap).resize((W, H), Image.NEAREST))
                cmap = plt.get_cmap("tab20", int(om.max()) + 1)
                ax.imshow(np.ma.masked_where(om == 0, om), cmap=cmap, alpha=.55, interpolation="nearest")
    names = ["Before fine-tuning", "After fine-tuning", "SAM masks\n(after fine-tuning)"][:nrows]
    for i, n in enumerate(names):
        axes[i, 0].text(-0.06, 0.5, n, transform=axes[i, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("Object detections with prompt 'part' (paper Fig. 2 style; the paper's 3rd row, hand-made labels, is not available)")
    fig.tight_layout()
    save(fig, out / "fig2_detections.png")


# ----------------------------------------------------------------------------------------- html
def html(report, run_dir):
    def table(p):
        rows = list(csv.reader(open(p)))
        h = "".join(f"<th>{c}</th>" for c in rows[0])
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
        return f"<div class='t'><table><tr>{h}</tr>{b}</table></div>"
    parts = [f"<h1>ObjectCore replication – {run_dir.name}</h1>",
             "<p>Reimplementation of ObjectCore (Fučka et al., WACV 2026) from the paper; MVTec LOCO few-shot protocol "
             "(k = 1, 2, 4 random support images, several draws). Paper numbers are from its Table 1 / Table 3.</p>"]
    for name, title in [("table1_comparison_long", "Paper Table 1 vs this replication"), ("ablation_comparison", "Paper Table 3 (ablation) vs this replication"),
                        ("per_category", "Per category"), ("context_fullshot", "Context: own full-shot replications")]:
        parts.append(f"<h2>{title}</h2>" + table(report / "tables" / f"{name}.csv"))
    for g in sorted((report / "graphs").glob("*.png")) + sorted((report / "figures").glob("*.png")) + sorted((report / "figures" / "per_category").glob("*.png")):
        parts.append(f"<h3>{g.stem}</h3><img src='{g.relative_to(report)}'>")
    css = ("body{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:16px}img{max-width:100%;border:1px solid #ddd}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;white-space:nowrap}.t{overflow-x:auto}")
    (report / "report.html").write_text(f"<!doctype html><meta charset='utf-8'><title>ObjectCore replication</title><style>{css}</style>" + "\n".join(parts))


# ----------------------------------------------------------------------------------------- main
def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "objectcore/runs/latest").resolve()
    data_root = Path(sys.argv[2] if len(sys.argv) > 2 else Path.home() / "objectcore/data/mvtec_loco")
    runs = load_runs(run_dir / "results")
    if not runs:
        sys.exit(f"no results found under {run_dir / 'results'}")
    report = run_dir / "report"
    for d in ["tables", "graphs", "figures"]:
        (report / d).mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs found in {run_dir}")
    make_tables(runs, report / "tables"); print("tables  ->", report / "tables")
    make_graphs(runs, report / "graphs"); print("graphs  ->", report / "graphs")
    detections(runs, data_root, report / "figures")
    for mode in ["best", "random"]:
        qualitative(runs, data_root, report / "figures", mode)
    per_category_examples(runs, data_root, report / "figures"); print("figures ->", report / "figures")
    html(report, run_dir); print("report  ->", report / "report.html")


if __name__ == "__main__":
    main()
