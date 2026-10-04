#!/usr/bin/env python3
"""Summarise ObjectCore replication results: per category and 5-category average, mean +- std over draws.

Averaging follows the paper's protocol: for each draw the 5 categories are averaged, then mean and std
are taken over the 3 draws. Writes summary.csv next to the results folder."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER_LOCO = {1: "72.5 +- 0.2", 2: "74.6 +- 0.6", 4: "80.8 +- 1.3"}
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
METRICS = ["all", "logical", "structural", "mean_LS"]

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "objectcore/runs/latest/results")
R = defaultdict(dict)  # (phase, k) -> {(cat, seed): auroc dict}
for f in root.glob("*/*/k*_seed*/results.json"):
    r = json.loads(f.read_text())
    R[(r["phase"], r["k"])][(r["category"], r["seed"])] = r

rows = []
for (phase, k) in sorted(R):
    runs = R[(phase, k)]
    seeds = sorted({s for _, s in runs})
    print(f"\n### {phase}  k={k}   (paper ObjectCore, LOCO I-AUROC: {PAPER_LOCO.get(k, '-')})")
    print(f"{'category':<20}" + "".join(f"{m:>18}" for m in METRICS) + "   branches all: struct-only / logic-only")
    for c in CATS:
        vals = {m: [runs[(c, s)]["auroc"][m] for s in seeds if (c, s) in runs] for m in METRICS}
        if not vals["all"]:
            continue
        bs = [runs[(c, s)]["auroc_structural_branch_only"]["all"] for s in seeds if (c, s) in runs]
        bl = [runs[(c, s)]["auroc_logical_branch_only"]["all"] for s in seeds if (c, s) in runs]
        print(f"{c:<20}" + "".join(f"{np.mean(v):>11.2f} +-{np.std(v):4.1f}" for v in vals.values())
              + f"   {np.mean(bs):.1f} / {np.mean(bl):.1f}   (n={len(vals['all'])})")
        rows.append([phase, k, c] + [f"{np.mean(vals[m]):.2f}" for m in METRICS] + [f"{np.std(vals['all']):.2f}", len(vals["all"])])
    # per-draw 5-category average, then mean +- std over draws
    complete = [s for s in seeds if all((c, s) in runs for c in CATS)]
    if complete:
        avg = {m: [np.mean([runs[(c, s)]["auroc"][m] for c in CATS]) for s in complete] for m in METRICS}
        print(f"{'AVERAGE (5 cats)':<20}" + "".join(f"{np.mean(v):>11.2f} +-{np.std(v):4.1f}" for v in avg.values())
              + f"   (draws={len(complete)})")
        rows.append([phase, k, "AVERAGE"] + [f"{np.mean(avg[m]):.2f}" for m in METRICS] + [f"{np.std(avg['all']):.2f}", len(complete)])
    else:
        print("AVERAGE: not all 5 categories finished yet")

with open(root / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["phase", "k", "category"] + [f"auroc_{m}" for m in METRICS] + ["std_all", "n_draws"])
    w.writerows(rows)
print(f"\nwrote {root / 'summary.csv'}")
print("Note: the paper does not state which I-AUROC definition it uses; compare 'all' and 'mean_LS'.")
