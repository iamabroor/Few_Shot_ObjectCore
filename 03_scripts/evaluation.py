#!/usr/bin/env python3
"""
Step 4 - evaluation.

  a) one scores file (from inference.py or a run folder):
       python 03_scripts/evaluation.py --scores ~/objectcore/models/breakfast_box_k4_s0/inference/scores.jsonl
     -> I-AUROC (all / logical / structural / mean L,S) and I-F1-max for the full score and each branch

  b) a whole run folder from run_all.sh (paper protocol):
       python 03_scripts/evaluation.py --run ~/objectcore/runs/2026-10-01_2301_all
     -> summarize.py tables (mean +- std over draws) and make_report.py (Table 1 / Table 3 comparison CSVs,
        graphs g1-g6, paper-style Fig. 2 / Fig. 3, report.html) in <run>/report/
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from _config import HERE
import objectcore as oc


def eval_scores(path):
    rows = [json.loads(l) for l in open(path)]
    lb, kd = [r["label"] for r in rows], [r["kind"] for r in rows]
    print(f"{path}: {len(rows)} images ({sum(lb)} anomalous)")
    for key, name in [("score", "ObjectCore (A_S + A_L)"), ("score_structural_only", "only M_S"),
                      ("score_logical_only", "only M_L")]:
        if key not in rows[0]:
            continue
        sc = [r[key] for r in rows]
        au = oc.aurocs(sc, lb, kd)
        print(f"  {name:24s} I-AUROC all {au['all']:6.2f} | logical {au['logical']:6.2f} | structural "
              f"{au['structural']:6.2f} | mean(L,S) {au['mean_LS']:6.2f} | I-F1-max {oc.f1_max(sc, lb):6.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scores")
    g.add_argument("--run")
    ap.add_argument("--data_root", default=str(Path.home() / "objectcore/data/mvtec_loco"))
    a = ap.parse_args()
    if a.scores:
        eval_scores(a.scores)
        return
    run = Path(a.run).expanduser()
    subprocess.run([sys.executable, str(HERE / "summarize.py"), str(run / "results")], check=True)
    subprocess.run([sys.executable, str(HERE / "make_report.py"), str(run), a.data_root], check=True)
    print(f"report: {run / 'report'}")


if __name__ == "__main__":
    main()
