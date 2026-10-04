#!/usr/bin/env python3
"""
Diagnose why detections vanish after fine-tuning (~5 min on the A10G).

    cd ~/objectcore && source env/bin/activate && export HF_HOME=~/objectcore/hf_cache
    python diag_finetune.py 2>&1 | grep -v -i warn | tee ~/objectcore/logs/diag_finetune.txt

Shows, for one breakfast_box support image (the k=1 seed=2 image from the main run):
  1. pretrained detector: how many queries pass 0.25 per prompt token ([CLS], part, ., [SEP])
  2. size of each loss term at step 0
  3. three fine-tuning variants (100 epochs each, fresh weights each time) and how many objects
     are detected afterwards on the support image and on 3 test images.
"""
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))  # flat copy in ~/objectcore
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_src" / "source_code"))  # repo layout
import objectcore as oc  # noqa: E402

ROOT = Path.home() / "objectcore"
DATA = ROOT / "data/mvtec_loco/breakfast_box"
dev = "cuda" if torch.cuda.is_available() else "cpu"
GD = sys.argv[1] if len(sys.argv) > 1 else "IDEA-Research/grounding-dino-base"
SAM = sys.argv[2] if len(sys.argv) > 2 else "facebook/sam-vit-base"
SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 800
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 100
if len(sys.argv) > 5:
    DATA = Path(sys.argv[5])

sup_path = DATA / "train/good/266.png"
if not sup_path.exists():
    sup_path = sorted((DATA / "train/good").glob("*.png"))[0]
test_paths = sorted((DATA / "test/good").glob("*.png"))[:3]
load = lambda p: oc.load_rgb(p)
sup = load(sup_path)
tests = [load(p) for p in test_paths]
print(f"device {dev} | support {sup_path.name} | test {[p.name for p in test_paths]}")

M = oc.Models(GD, SAM, dev)
tok = M.gproc.tokenizer
ids = M.g_inputs(sup, SIZE)["input_ids"][0].tolist()
names = tok.convert_ids_to_tokens(ids)
print("prompt tokens:", list(zip(ids, names)))


@torch.no_grad()
def token_stats(img, tag):
    inp = M.g_inputs(img, SIZE)
    out = M.gdino(**inp)
    p = out.logits[0, :, : len(ids)].sigmoid()  # 900 x n_tokens
    parts = [f"{n}: max {p[:, i].max():.2f}, >0.25 in {int((p[:, i] > 0.25).sum())} queries" for i, n in enumerate(names)]
    dets = len(M.detect(img, SIZE, 0.25, 0.25)[0])
    print(f"  [{tag}] detections {dets:3d} | " + " | ".join(parts))
    return dets


def n_dets(img):
    return len(M.detect(img, SIZE, 0.25, 0.25)[0])


print("\n== 1. Pretrained detector (no fine-tuning)")
token_stats(sup, "support")
pseudo = M.detect(sup, SIZE, 0.25, 0.25)[0]
print(f"  pseudo-labels on support image: {len(pseudo)} boxes; test images: {[n_dets(t) for t in tests]}")

print("\n== 2. Loss terms at step 0 (all prompt tokens vs phrase token only)")
W, H = sup.size
b = pseudo.float()
cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                      (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H], 1).clamp(0, 1)
labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long, device=dev), "boxes": cxcywh.to(dev)}]
M.gdino.config.bbox_loss_coefficient = 0.0
M.gdino.config.giou_loss_coefficient = 0.0
M.gdino.config.auxiliary_loss = True
for mode in ["all", "phrase"]:
    with oc.phrase_token_loss(mode == "phrase") as ctx, torch.no_grad():
        inp = M.g_inputs(sup, SIZE)
        ctx.input_ids = inp["input_ids"]
        out = M.gdino(**inp, labels=labels)
        ce = {k: round(v.item(), 1) for k, v in out.loss_dict.items() if k.startswith("loss_ce")}
        print(f"  {mode:6s}: total weighted loss {out.loss.item():.1f} | loss_ce terms {ce}")

variants = [
    ("V1 current: all tokens, clip 0.1", dict(ft_loss_tokens="all", ft_clip=0.1)),
    ("V2 phrase token only, clip 0.1", dict(ft_loss_tokens="phrase", ft_clip=0.1)),
    ("V3 phrase token only, no clipping", dict(ft_loss_tokens="phrase", ft_clip=0.0)),
]
print(f"\n== 3. Fine-tuning variants ({EPOCHS} epochs on the 1 support image, lr 1e-5, AdamW)")
summary = []
for name, kw in variants:
    M.reload_detector()
    args = types.SimpleNamespace(img_size=SIZE, ft_lr=1e-5, ft_wd=1e-4, ft_epochs=EPOCHS, **kw)
    t0 = time.time()
    oc.finetune_detector(M, [sup], [pseudo], args, lambda s: print("   ", s.strip()))
    print(f"  {name}  ({time.time() - t0:.0f}s)")
    d_sup = token_stats(sup, "support after")
    d_test = [n_dets(t) for t in tests]
    print(f"  test images after: {d_test}")
    summary.append((name, len(pseudo), d_sup, d_test))

print("\n== SUMMARY  (objects detected at 0.25; pseudo-labels on support = target)")
for name, n_pseudo, d_sup, d_test in summary:
    print(f"  {name:38s} support {d_sup:3d} (pseudo {n_pseudo}) | test {d_test}")
print("Send this summary to Claude.")
