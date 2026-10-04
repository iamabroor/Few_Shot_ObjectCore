#!/usr/bin/env python3
"""
Step 1 - data preparation for ObjectCore on MVTec LOCO.

  * checks the dataset layout and counts images per split (train/good, test good / logical / structural, GT masks)
  * writes the random support draws used by the paper protocol (k = 1, 2, 4 x 3 draws), exactly as
    objectcore.py draws them: random.Random(1000 * k + seed).sample(train_good, k)

The dataset itself is downloaded by setup.sh (MVTec LOCO, CC BY-NC-SA 4.0 - not redistributed in this repo).

    python 03_scripts/preprocessing.py --config phase_B_phrase_fix.yaml --out support_draws.csv
"""
import argparse
import csv
import random
from pathlib import Path

from _config import load_config
import objectcore as oc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="objectcore_default.yaml")
    ap.add_argument("--data_root", default=None, help="overrides data_root in the config")
    ap.add_argument("--out", default="support_draws.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.data_root or cfg["data_root"]).expanduser()
    if not (root / "breakfast_box").exists():
        raise SystemExit(f"MVTec LOCO not found under {root}. Run 03_scripts/setup.sh (step 2/4 downloads it).")

    print(f"{'category':22s} {'train':>6s} {'good':>5s} {'logic':>6s} {'struct':>7s} {'GT dirs':>8s}")
    rows = []
    for cat in cfg["categories"]:
        train, test = oc.load_category(root, cat)
        n = {k: sum(1 for _, _, kd in test if kd == k) for k in ("good", "logical", "structural")}
        gt = sum(1 for d in (root / cat / "ground_truth").glob("*/*") if d.is_dir())
        print(f"{cat:22s} {len(train):6d} {n['good']:5d} {n['logical']:6d} {n['structural']:7d} {gt:8d}")
        seeds = cfg.get("draw_seeds") or list(range(cfg["draws"]))
        for k in cfg["shots"]:
            for s in seeds:
                sup = random.Random(1000 * k + s).sample(train, k)
                rows.append({"category": cat, "k": k, "seed": s, "support_images": " ".join(p.name for p in sup)})

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} support draws written to {a.out}")


if __name__ == "__main__":
    main()
