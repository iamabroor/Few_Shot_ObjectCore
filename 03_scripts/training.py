#!/usr/bin/env python3
"""
Step 2 - "training" for ONE category / k / support draw.

ObjectCore has no trained network of its own. Training means:
  1. draw k normal support images (same draw as the paper protocol: random.Random(1000*k + seed))
  2. GroundingDINO (prompt "part") pseudo-detections on the support images
  3. Phase B only: fine-tune GroundingDINO on those pseudo-boxes (L_cls only, 100 epochs, AdamW, lr 1e-5)
  4. SAM masks + Swin features -> structural memory bank M_S and logical memory bank M_L
The memory banks are saved to <out>/model.pt; with --save_detector the fine-tuned detector is saved too
(~900 MB), so inference.py can reuse it without fine-tuning again.

    python 03_scripts/training.py --config phase_B_phrase_fix.yaml --category breakfast_box --k 4 --seed 0 \
        --out ~/objectcore/models/breakfast_box_k4_s0 --save_detector

The full paper protocol (5 categories x k = 1, 2, 4 x 3 draws, training + inference + metrics in one go)
is 03_scripts/run_all.sh, which calls objectcore.py directly.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from _config import load_config, to_namespace
import objectcore as oc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="phase_B_phrase_fix.yaml")
    ap.add_argument("--category", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--gdino", default=None)
    ap.add_argument("--sam", default=None)
    ap.add_argument("--out", required=True, help="output folder for model.pt (+ detector/)")
    ap.add_argument("--save_detector", action="store_true", help="also save the fine-tuned GroundingDINO")
    a = ap.parse_args()
    cfg = load_config(a.config)
    args = to_namespace(cfg, data_root=a.data_root, gdino=a.gdino, sam=a.sam)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train, _ = oc.load_category(args.data_root, a.category)
    sup_paths = random.Random(1000 * a.k + a.seed).sample(train, a.k)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    print(f"{a.category} k={a.k} seed={a.seed} support: {[p.name for p in sup_paths]} | finetune={args.finetune}")
    sup_imgs = [oc.load_rgb(p) for p in sup_paths]
    M = oc.Models(args.gdino, args.sam, dev, args.prompt)

    t0 = time.time()
    pseudo = [M.detect(im, args.img_size, args.box_threshold, args.text_threshold, args.nms)[0] for im in sup_imgs]
    print(f"pseudo-detections per support image: {[len(b) for b in pseudo]}")
    if args.finetune:
        oc.finetune_detector(M, sup_imgs, pseudo, args, print)
    recs = [oc.describe_image(M, im, args) for im in sup_imgs]
    print(f"objects per support image after {'fine-tuning' if args.finetune else 'detection'}: "
          f"{[len(r['objs']) for r in recs]}")

    torch.save({"support_recs": recs,
                "support_images": [str(p.relative_to(Path(args.data_root))) for p in sup_paths],
                "pseudo_detections": [len(b) for b in pseudo],
                "config": vars(args), "category": a.category, "k": a.k, "seed": a.seed},
               out / "model.pt")
    if args.finetune and a.save_detector:
        M.gdino.save_pretrained(out / "detector")
        M.gproc.save_pretrained(out / "detector")
    info = {"category": a.category, "k": a.k, "seed": a.seed, "finetune": args.finetune,
            "ft_loss_tokens": args.ft_loss_tokens,
            "support_images": [p.name for p in sup_paths],
            "pseudo_detections": [len(b) for b in pseudo],
            "objects_per_support": [len(r["objs"]) for r in recs],
            "structural_bank_size": int(sum(r["feat"].shape[1] * r["feat"].shape[2] for r in recs)),
            "detector_saved": bool(args.finetune and a.save_detector),
            "time_s": round(time.time() - t0, 1)}
    json.dump(info, open(out / "training_info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
