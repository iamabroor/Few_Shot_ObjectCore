#!/usr/bin/env python3
"""
Step 3 - inference: score test images with a model folder written by training.py.

  per test image: detect objects (fine-tuned detector if saved, otherwise the config's detector),
  SAM masks, Swin features -> A_S (nearest-neighbour patch distance to M_S) and A_L (Hungarian object
  matching against the most similar support image) -> A = Gauss_sigma4(A_S + A_L) -> score = mean of top 1 %.

    python 03_scripts/inference.py --model ~/objectcore/models/breakfast_box_k4_s0              # all test images
    python 03_scripts/inference.py --model ... --images test/logical_anomalies/000.png --png     # single image + heat map

Writes <model>/inference/scores.jsonl and, with --save_maps / --png, anomaly maps (npz / png).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _config import to_namespace
import objectcore as oc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder written by training.py")
    ap.add_argument("--images", nargs="*", default=None, help="paths relative to the category folder (default: all test)")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--save_maps", action="store_true")
    ap.add_argument("--png", action="store_true", help="save image | A_S | A_L | A as png per image")
    ap.add_argument("--out", default=None, help="default <model>/inference")
    a = ap.parse_args()
    mdir = Path(a.model).expanduser()
    ck = torch.load(mdir / "model.pt", map_location="cpu", weights_only=False)
    args = to_namespace(ck["config"], data_root=a.data_root)
    cat = ck["category"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out).expanduser() if a.out else mdir / "inference"
    out.mkdir(parents=True, exist_ok=True)

    gdino = str(mdir / "detector") if (mdir / "detector").exists() else args.gdino
    if args.finetune and gdino == args.gdino:
        print("NOTE: model was trained with fine-tuning but no detector/ was saved -> using the pretrained "
              "detector for test images (re-run training.py with --save_detector to match the paper).")
    M = oc.Models(gdino, args.sam, dev, args.prompt)
    model = oc.ObjectCore(ck["support_recs"], args, dev)

    _, test = oc.load_category(args.data_root, cat)
    if a.images:
        want = {str(Path(i)) for i in a.images}
        test = [t for t in test if str(t[0].relative_to(Path(args.data_root) / cat)) in want]
        if not test:
            raise SystemExit(f"none of {a.images} found under {Path(args.data_root) / cat}")
    rows = []
    with open(out / "scores.jsonl", "w") as f:
        for i, (p, lab, kind) in enumerate(test):
            rec = oc.describe_image(M, oc.load_rgb(p), args)
            s, s_s, s_l, nn, maps = model.score(rec)
            rel = str(p.relative_to(Path(args.data_root)))
            row = {"image": rel, "label": lab, "kind": kind, "score": s,
                   "score_structural_only": s_s, "score_logical_only": s_l,
                   "n_objects": len(rec["objs"]), "nn_support": ck["support_images"][nn]}
            f.write(json.dumps(row) + "\n")
            rows.append(row)
            stem = rel.replace("/", "_").rsplit(".", 1)[0]
            if a.save_maps:
                np.savez_compressed(out / f"{stem}.npz", **{k: v.cpu().numpy().astype(np.float16) for k, v in maps.items()})
            if a.png:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
                ax[0].imshow(oc.load_rgb(p)); ax[0].set_title(f"{kind}  score {s:.3f}")
                for j, (key, t) in enumerate([("AS", "A_S structural"), ("AL", "A_L logical"), ("A", "A = Gauss(A_S + A_L)")], 1):
                    ax[j].imshow(maps[key].cpu().numpy(), cmap="jet"); ax[j].set_title(t)
                for x in ax:
                    x.axis("off")
                fig.tight_layout(); fig.savefig(out / f"{stem}.png", dpi=150); plt.close(fig)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(test)}")
    print(f"{len(rows)} images scored -> {out / 'scores.jsonl'}")
    if len({r['label'] for r in rows}) == 2:
        sc, lb, kd = [r["score"] for r in rows], [r["label"] for r in rows], [r["kind"] for r in rows]
        au = oc.aurocs(sc, lb, kd)
        print(f"I-AUROC all {au['all']:.2f} | logical {au['logical']:.2f} | structural {au['structural']:.2f} | "
              f"I-F1-max {oc.f1_max(sc, lb):.2f}")


if __name__ == "__main__":
    main()
