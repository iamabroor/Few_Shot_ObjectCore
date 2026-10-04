#!/usr/bin/env python3
"""
ObjectCore (Fučka, Zavrtanik, Skočaj - WACV 2026) - independent reimplementation from the paper.

The official code (github.com/MaticFuc/ObjectCore) is not public, so this file rebuilds the
method from the paper + supplementary material. Every setting the paper states is used as stated;
every setting the paper leaves open is a command-line flag and is listed in IMPLEMENTATION_CHOICES
(written to each run's results.json so it ends up in the replication record).

Pipeline per category, per support draw (k random train/good images):
  1. GroundingDINO (Swin-B), prompt "part", box & text thresholds 0.25, images 800x800
     -> pseudo-detections on the support images.
  2. (Phase B, --finetune) fine-tune GroundingDINO on the support images with the pseudo-boxes,
     classification loss only (Lbbox/GIoU weights = 0), 100 epochs, AdamW lr 1e-5, batch 1.
  3. Detect objects (support + test) with the (fine-tuned) detector, SAM ViT-B mask per box.
  4. Patch features = first three Swin stages of GroundingDINO's image encoder, 5x5 mean filter.
     Object representation o = mean feature inside its SAM mask.
  5. Structural memory bank M_S = all support patches (no coreset); A_S = NN cosine distance.
  6. Logical memory bank M_L = {mean image feature : object set}. For a test image, take the most
     similar support image, Hungarian-match objects with cost cos(o_t,o_s) + 0.1*|size_t - size_s|,
     pad with max cost. A_L = sum matched m_t*cost + sum unmatched test m_t + sum unmatched support m_s.
  7. A = Gaussian(sigma=4) * (A_S + A_L); image score = mean of top 1% of A.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

IMPLEMENTATION_CHOICES = {
    "feature_layers": "Swin stage1, stage2, stage3 outputs (strides 4/8/16), taken before patch merging",
    "feature_combination": "each stage 5x5 mean-filtered at native resolution, L2-normalised per patch, "
                           "resized to a common grid (--feat_res), concatenated",
    "mask_size_units": "fraction of image area (mask pixels / all pixels)",
    "map_combination": "A_S + A_L added without rescaling, both on the feature grid, then upsampled to "
                       "--map_res and Gaussian-smoothed with sigma=4 at that resolution",
    "pad_cost": "max cosine distance (2) + gamma * 1",
    "detector_nms": "none (paper mentions none) unless --nms is set",
    "finetune_trainable": "all GroundingDINO parameters (paper does not say what is frozen)",
    "finetune_weight_decay": "1e-4 (GroundingDINO default; paper does not say)",
    "finetune_aux_loss": "decoder auxiliary + encoder (two-stage) classification losses on, box losses weighted 0",
    "auroc": "reported 4 ways: all (good vs all anomalies), logical, structural, mean(logical, structural); "
             "'all' is compared with the paper's Table 1",
    "pixel_metrics": "P-AUROC / P-F1-max over all test pixels at 256x256 (maps and GT both resized to 256x256, "
                     "GT = union of all LOCO GT masks of an image)",
}


# ----------------------------------------------------------------------------------------- data
def list_images(d):
    d = Path(d)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def load_category(root, cat):
    root = Path(root) / cat
    train = list_images(root / "train" / "good")
    test = []
    for sub, label, kind in [("good", 0, "good"),
                             ("logical_anomalies", 1, "logical"),
                             ("structural_anomalies", 1, "structural")]:
        for p in list_images(root / "test" / sub):
            test.append((p, label, kind))
    if not train or not test:
        raise FileNotFoundError(f"No images found under {root} (expected train/good and test/<good|logical_anomalies|structural_anomalies>)")
    return train, test


def load_rgb(p):
    return Image.open(p).convert("RGB")


# ------------------------------------------------------------------------------------- models
class Models:
    def __init__(self, gdino_path, sam_path, device, prompt="part."):
        self.device = device
        self.prompt = prompt
        self.gproc = AutoProcessor.from_pretrained(gdino_path)
        self.gdino_path = gdino_path
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_path).to(device).eval()
        self.sproc = SamProcessor.from_pretrained(sam_path)
        self.sam = SamModel.from_pretrained(sam_path).to(device).eval()

    def reload_detector(self):
        del self.gdino
        torch.cuda.empty_cache()
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_path).to(self.device).eval()

    def g_inputs(self, img, size):
        inp = self.gproc(images=img, text=self.prompt, return_tensors="pt",
                         size={"height": size, "width": size})
        return {k: v.to(self.device) for k, v in inp.items()}

    # --- detection ---------------------------------------------------------------------------
    @torch.no_grad()
    def detect(self, img, size, box_thr, text_thr, nms_iou=None):
        inp = self.g_inputs(img, size)
        out = self.gdino(**inp)
        res = self.gproc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=box_thr, text_threshold=text_thr,
            target_sizes=[(img.height, img.width)])[0]
        boxes, scores = res["boxes"].float(), res["scores"].float()
        if nms_iou is not None and len(boxes) > 1:
            from torchvision.ops import nms
            keep = nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep], scores[keep]
        # clip + drop degenerate boxes
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img.width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img.height)
        ok = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
        return boxes[ok].cpu(), scores[ok].cpu()

    # --- features: first three Swin stages of GroundingDINO's image encoder --------------------
    @torch.no_grad()
    def features(self, img, size, feat_res, mean_k):
        inp = self.g_inputs(img, size)
        bb = self.gdino.model.backbone.conv_encoder.model  # SwinBackbone
        emb, dims = bb.embeddings(inp["pixel_values"])
        enc = bb.encoder(emb, dims, output_hidden_states=True,
                         output_hidden_states_before_downsampling=True,
                         always_partition=True, return_dict=True)
        stages = enc.reshaped_hidden_states[1:4]  # index 0 = patch embeddings ("stem")
        feats = []
        for f in stages:
            f = f.float()
            if mean_k > 1:
                f = F.avg_pool2d(f, mean_k, stride=1, padding=mean_k // 2, count_include_pad=False)
            f = F.normalize(f, dim=1)
            f = F.interpolate(f, size=(feat_res, feat_res), mode="bilinear", align_corners=False)
            feats.append(f)
        f = torch.cat(feats, dim=1)[0]  # C x H x W
        return f

    # --- SAM masks from boxes ----------------------------------------------------------------
    @torch.no_grad()
    def masks(self, img, boxes, chunk=32):
        if len(boxes) == 0:
            return torch.zeros(0, img.height, img.width, dtype=torch.bool)
        out_masks = []
        for i in range(0, len(boxes), chunk):
            b = boxes[i:i + chunk].tolist()
            inp = self.sproc(img, input_boxes=[b], return_tensors="pt").to(self.device)
            out = self.sam(**inp, multimask_output=False)
            m = self.sproc.image_processor.post_process_masks(
                out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
            out_masks.append(m[:, 0].bool())
        return torch.cat(out_masks, 0)


# ----------------------------------------------------------------------------- per-image record
def describe_image(models, img, args):
    """Patch features, objects (representation, mask on feature grid, mask size)."""
    feat = models.features(img, args.img_size, args.feat_res, args.mean_kernel)  # C x h x w on device
    boxes, box_scores = models.detect(img, args.img_size, args.box_threshold, args.text_threshold, args.nms)
    masks = models.masks(img, boxes)  # N x H x W bool (original resolution)
    objs, grid_masks, sizes = [], [], []
    for m in masks:
        area = m.float().mean().item()
        if area <= 0:
            continue
        gm = F.interpolate(m[None, None].float(), size=feat.shape[-2:], mode="area")[0, 0].to(feat.device)
        if gm.sum() <= 0:
            continue
        o = (feat * gm).sum(dim=(1, 2)) / gm.sum()
        objs.append(F.normalize(o, dim=0))
        grid_masks.append(gm)
        sizes.append(area)
    C = feat.shape[0]
    rec = {
        "feat": feat.half().cpu(),
        "gfeat": F.normalize(feat.mean(dim=(1, 2)), dim=0).cpu(),
        "objs": torch.stack(objs).cpu() if objs else torch.zeros(0, C),
        "masks": torch.stack(grid_masks).half().cpu() if grid_masks else torch.zeros(0, *feat.shape[-2:]).half(),
        "sizes": torch.tensor(sizes, dtype=torch.float32),
        "n_boxes": int(len(boxes)),
        # for figures: boxes normalised to [0,1] (x0,y0,x1,y1) + detection scores
        "boxes_norm": (boxes / torch.tensor([img.width, img.height, img.width, img.height])).tolist() if len(boxes) else [],
        "box_scores": box_scores.tolist() if len(boxes) else [],
    }
    return rec


# ---------------------------------------------------------------------------------- scoring
def gaussian_kernel(sigma):
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(a, sigma):
    k = gaussian_kernel(sigma).to(a.device)
    a = a[None, None]
    r = len(k) // 2
    a = F.conv2d(F.pad(a, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    a = F.conv2d(F.pad(a, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return a[0, 0]


class ObjectCore:
    def __init__(self, support_recs, args, device):
        self.args, self.device = args, device
        # structural memory bank: every support patch (no coreset)
        self.MS = torch.cat([r["feat"].flatten(1).T for r in support_recs], 0).to(device).float()
        self.MS = F.normalize(self.MS, dim=1)
        # logical memory bank: mean image feature -> object set
        self.keys = torch.stack([r["gfeat"] for r in support_recs]).to(device)
        self.support = support_recs

    @torch.no_grad()
    def structural_map(self, feat):
        C, h, w = feat.shape
        q = F.normalize(feat.flatten(1).T.float().to(self.device), dim=1)
        d = []
        for i in range(0, q.shape[0], 4096):
            sim = q[i:i + 4096] @ self.MS.T
            d.append(1 - sim.max(dim=1).values)
        return torch.cat(d).view(h, w)

    @torch.no_grad()
    def logical_map(self, rec):
        g = self.args.gamma
        h, w = rec["feat"].shape[-2:]
        idx = int((self.keys @ rec["gfeat"].to(self.device)).argmax())
        s = self.support[idx]
        A = torch.zeros(h, w, device=self.device)
        ot, os_ = rec["objs"].to(self.device), s["objs"].to(self.device)
        mt, ms = rec["masks"].float().to(self.device), s["masks"].float().to(self.device)
        st, ss = rec["sizes"].to(self.device), s["sizes"].to(self.device)
        nt, ns = len(ot), len(os_)
        if nt == 0 and ns == 0:
            return A, idx
        n = max(nt, ns)
        pad = 2.0 + g * 1.0
        cost = torch.full((n, n), pad, device=self.device)
        if nt and ns:
            cost[:nt, :ns] = (1 - ot @ os_.T) + g * (st[:, None] - ss[None, :]).abs()
        r, c = linear_sum_assignment(cost.cpu().numpy())
        for i, j in zip(r, c):
            if i < nt and j < ns:  # matched pair
                A += mt[i] * cost[i, j]
            elif i < nt:  # unmatched test object
                A += mt[i]
            elif j < ns:  # unmatched support object
                A += ms[j]
        return A, idx

    @torch.no_grad()
    def score(self, rec):
        AS = self.structural_map(rec["feat"])
        AL, nn_idx = self.logical_map(rec)
        A = (AS + AL)[None, None]
        A = F.interpolate(A, size=(self.args.map_res, self.args.map_res), mode="bilinear", align_corners=False)[0, 0]
        A = gaussian_blur(A, self.args.sigma)
        flat = A.flatten()
        k = max(1, int(round(self.args.top_frac * flat.numel())))
        s = flat.topk(k).values.mean().item()
        # branch-only maps/scores (for analysis + figures, same smoothing and aggregation)
        def smooth(m):
            m = F.interpolate(m[None, None], size=(self.args.map_res,) * 2, mode="bilinear", align_corners=False)[0, 0]
            return gaussian_blur(m, self.args.sigma)
        AS_s, AL_s = smooth(AS), smooth(AL)
        agg = lambda m: m.flatten().topk(k).values.mean().item()
        return s, agg(AS_s), agg(AL_s), nn_idx, {"A": A, "AS": AS_s, "AL": AL_s}


# ------------------------------------------------------------------------------ fine-tuning
SPECIAL_TOKEN_IDS = (101, 102, 1012, 1029)  # [CLS], [SEP], ".", "?" (BERT ids, as in GroundingDINO)


class phrase_token_loss:
    """Context manager: restrict GroundingDINO's classification loss to the prompt's phrase tokens
    (e.g. "part"), i.e. leave [CLS] / "." / [SEP] out of the loss (--ft_loss_tokens phrase)."""
    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self
        import transformers.models.grounding_dino.modeling_grounding_dino as gd
        self.gd, self.orig = gd, gd.build_text_mask
        state = self

        def build_text_mask(logits, attention_mask):
            mask = state.orig(logits, attention_mask)
            ids = state.input_ids
            special = torch.zeros(mask.shape[-1], dtype=torch.bool, device=mask.device)
            for t in SPECIAL_TOKEN_IDS:
                special[: ids.shape[1]] |= (ids[0] == t)
            return mask & ~special[None, None, :]
        gd.build_text_mask = build_text_mask
        return self

    def __exit__(self, *a):
        if self.enabled:
            self.gd.build_text_mask = self.orig


def finetune_detector(models, support_imgs, pseudo, args, log):
    """Phase B: classification loss only (GLIP-style contrastive focal loss), Lbbox/GIoU weights = 0."""
    with phrase_token_loss(getattr(args, "ft_loss_tokens", "all") == "phrase") as ctx:
        return _finetune_detector(models, support_imgs, pseudo, args, log, ctx)


def _finetune_detector(models, support_imgs, pseudo, args, log, ctx):
    m = models.gdino
    m.config.bbox_loss_coefficient = 0.0
    m.config.giou_loss_coefficient = 0.0
    m.config.auxiliary_loss = True
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.ft_lr, weight_decay=args.ft_wd)
    data = [(img, b) for img, b in zip(support_imgs, pseudo) if len(b) > 0]
    if not data:
        log("  [finetune] no pseudo-detections on any support image - skipping fine-tuning")
        m.eval()
        return
    t0 = time.time()
    for ep in range(args.ft_epochs):
        random.shuffle(data)
        tot = 0.0
        for img, boxes in data:
            inp = models.g_inputs(img, args.img_size)
            W, H = img.width, img.height
            b = boxes.clone().float()
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / W, (b[:, 1] + b[:, 3]) / 2 / H,
                                  (b[:, 2] - b[:, 0]) / W, (b[:, 3] - b[:, 1]) / H], 1).clamp(0, 1)
            labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long, device=models.device),
                       "boxes": cxcywh.to(models.device)}]
            ctx.input_ids = inp["input_ids"]
            out = m(**inp, labels=labels)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            if getattr(args, "ft_clip", 0.1) > 0:
                torch.nn.utils.clip_grad_norm_(params, getattr(args, "ft_clip", 0.1))
            opt.step()
            tot += out.loss.item()
        if ep == 0 or (ep + 1) % 10 == 0:
            log(f"  [finetune] epoch {ep + 1}/{args.ft_epochs} loss {tot / len(data):.4f} ({time.time() - t0:.0f}s)")
    m.eval()


# ---------------------------------------------------------------------------------- metrics
def aurocs(scores, labels, kinds):
    scores, labels, kinds = np.asarray(scores), np.asarray(labels), np.asarray(kinds)
    good = kinds == "good"
    out = {"all": roc_auc_score(labels, scores) * 100}
    for k in ["logical", "structural"]:
        sel = good | (kinds == k)
        out[k] = roc_auc_score(labels[sel], scores[sel]) * 100 if (kinds == k).any() else float("nan")
    out["mean_LS"] = (out["logical"] + out["structural"]) / 2
    return out


def f1_max(scores, labels):
    """Max F1 over all thresholds (I-F1-max / P-F1-max as in the paper's Table 1)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # only evaluate at the last index of each run of tied scores
    last = np.r_[np.diff(scores[order]) != 0, True]
    tp, fp = tp[last], fp[last]
    P = labels.sum()
    if P == 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(f1.max() * 100)


def load_gt(data_root, rel_image, kind, size):
    """MVTec LOCO GT: ground_truth/<logical|structural>_anomalies/<image id>/*.png (union of all masks)."""
    if kind == "good":
        return np.zeros((size, size), dtype=bool)
    p = Path(data_root) / rel_image
    sub = "logical_anomalies" if kind == "logical" else "structural_anomalies"
    d = p.parents[2] / "ground_truth" / sub / p.stem
    m = None
    for f in sorted(d.glob("*.png")) if d.exists() else []:
        g = np.array(Image.open(f).convert("L").resize((size, size), Image.NEAREST)) > 0
        m = g if m is None else (m | g)
    if m is None:
        return None  # GT missing
    return m


def pixel_metrics(maps, gts):
    """P-AUROC and P-F1-max over all test pixels (images without GT files are skipped)."""
    pairs = [(m, g) for m, g in zip(maps, gts) if g is not None]
    if not pairs:
        return {"P-AUROC": float("nan"), "P-F1-max": float("nan")}
    sc = np.concatenate([m.ravel() for m, _ in pairs]).astype(np.float32)
    gt = np.concatenate([g.ravel() for _, g in pairs])
    if gt.all() or not gt.any():
        return {"P-AUROC": float("nan"), "P-F1-max": float("nan")}
    return {"P-AUROC": float(roc_auc_score(gt, sc) * 100), "P-F1-max": f1_max(sc, gt)}


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="MVTec LOCO root (contains breakfast_box/, ...)")
    ap.add_argument("--category", required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--draws", type=int, default=3, help="random support draws per k (paper: 3)")
    ap.add_argument("--draw_seeds", type=int, nargs="*", default=None, help="explicit seeds, default 0..draws-1")
    ap.add_argument("--finetune", action="store_true", help="Phase B: fine-tune detector on the support set")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--prompt", default="part.")
    # paper settings
    ap.add_argument("--img_size", type=int, default=800)
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--mean_kernel", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--top_frac", type=float, default=0.01)
    ap.add_argument("--ft_epochs", type=int, default=100)
    ap.add_argument("--ft_lr", type=float, default=1e-5)
    # open choices
    ap.add_argument("--feat_res", type=int, default=100, help="common feature grid (100 = stride 8 at 800px)")
    ap.add_argument("--map_res", type=int, default=256)
    ap.add_argument("--ft_wd", type=float, default=1e-4)
    ap.add_argument("--ft_loss_tokens", choices=["all", "phrase"], default="all",
                    help="tokens in the fine-tuning classification loss: all prompt tokens (HF default) or only the phrase ('part')")
    ap.add_argument("--ft_clip", type=float, default=0.1, help="gradient-norm clipping in fine-tuning (GroundingDINO default 0.1; 0 = off)")
    ap.add_argument("--nms", type=float, default=None, help="optional NMS IoU (off by default)")
    ap.add_argument("--max_test", type=int, default=None, help="debug: limit number of test images")
    ap.add_argument("--save_maps", nargs="*", default=["4:0"],
                    help="k:seed pairs whose anomaly maps are saved for figures (default 4:0); 'none' to disable")
    ap.add_argument("--pixel_res", type=int, default=256, help="resolution for pixel metrics (maps and GT)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase = "B_finetune" if args.finetune else "A_zeroshot"
    out_root = Path(args.out_dir) / phase / args.category
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    train, test = load_category(args.data_root, args.category)
    if args.max_test:
        rng = random.Random(0)
        test = rng.sample(test, min(args.max_test, len(test)))
    log(f"{args.category}: {len(train)} train/good, {len(test)} test | phase {phase} | device {device}")

    models = Models(args.gdino, args.sam, device, args.prompt)
    seeds = args.draw_seeds if args.draw_seeds else list(range(args.draws))

    test_cache = None  # Phase A: test descriptions don't depend on the support set -> compute once

    for k in args.shots:
        for seed in seeds:
            run_dir = out_root / f"k{k}_seed{seed}"
            if (run_dir / "results.json").exists():
                log(f"skip k={k} seed={seed} (done)")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            rng = random.Random(1000 * k + seed)
            sup_paths = rng.sample(train, k)
            log(f"k={k} seed={seed} support: {[p.name for p in sup_paths]}")
            torch.manual_seed(seed)
            random.seed(seed)
            sup_imgs = [load_rgb(p) for p in sup_paths]

            if args.finetune:
                models.reload_detector()  # fresh pretrained weights for every draw
                pseudo = [models.detect(im, args.img_size, args.box_threshold, args.text_threshold, args.nms)[0]
                          for im in sup_imgs]
                log(f"  pseudo-detections per support image: {[len(b) for b in pseudo]}")
                finetune_detector(models, sup_imgs, pseudo, args, log)

            sup_recs = [describe_image(models, im, args) for im in sup_imgs]
            log(f"  objects per support image: {[len(r['objs']) for r in sup_recs]}")
            oc = ObjectCore(sup_recs, args, device)

            if args.finetune or test_cache is None:
                cache = []
                for i, (p, lab, kind) in enumerate(test):
                    cache.append(describe_image(models, load_rgb(p), args))
                    if (i + 1) % 50 == 0:
                        log(f"  described {i + 1}/{len(test)} test images")
                if not args.finetune:
                    test_cache = cache
            else:
                cache = test_cache

            save_maps = f"{k}:{seed}" in (args.save_maps or [])
            if save_maps:
                (run_dir / "maps").mkdir(exist_ok=True)
            rows, S, SA, SL = [], [], [], []
            pmaps = {"A": [], "AS": [], "AL": []}
            gts = []
            for i, ((p, lab, kind), rec) in enumerate(zip(test, cache)):
                s, s_s, s_l, nn_idx, maps = oc.score(rec)
                S.append(s); SA.append(s_s); SL.append(s_l)
                rel = str(p.relative_to(Path(args.data_root)))
                for key in pmaps:
                    m = maps[key]
                    if m.shape[-1] != args.pixel_res:
                        m = F.interpolate(m[None, None], size=(args.pixel_res,) * 2, mode="bilinear", align_corners=False)[0, 0]
                    pmaps[key].append(m.cpu().numpy().astype(np.float16))
                gts.append(load_gt(args.data_root, rel, kind, args.pixel_res))
                rows.append({"image": rel, "label": lab, "kind": kind,
                             "score": s, "score_structural_only": s_s, "score_logical_only": s_l,
                             "n_objects": len(rec["objs"]), "n_boxes": rec["n_boxes"],
                             "nn_support": sup_paths[nn_idx].name,
                             "boxes_norm": [[round(v, 4) for v in b] for b in rec["boxes_norm"]],
                             "box_scores": [round(v, 3) for v in rec["box_scores"]]})
                if save_maps:
                    m = rec["masks"].float()
                    objmap = (m.argmax(0) + 1) * (m.max(0).values > 0.5) if len(m) else torch.zeros(m.shape[-2:])
                    np.savez_compressed(run_dir / "maps" / f"{i:04d}.npz",
                                        A=pmaps["A"][-1], AS=pmaps["AS"][-1], AL=pmaps["AL"][-1],
                                        objmap=objmap.numpy().astype(np.uint8),
                                        boxes=np.array(rec["boxes_norm"], dtype=np.float32).reshape(-1, 4),
                                        image=rel, kind=kind)
            labels = [r["label"] for r in rows]
            kinds = [r["kind"] for r in rows]
            n_gt_missing = sum(1 for g, kd in zip(gts, kinds) if g is None)
            metrics = {"I-AUROC": aurocs(S, labels, kinds)["all"], "I-F1-max": f1_max(S, labels)}
            metrics.update(pixel_metrics(pmaps["A"], gts))
            branch_metrics = {}
            for key, sc in [("structural_only", SA), ("logical_only", SL)]:
                bm = {"I-AUROC": aurocs(sc, labels, kinds)["all"], "I-F1-max": f1_max(sc, labels)}
                bm.update(pixel_metrics(pmaps["AS" if key == "structural_only" else "AL"], gts))
                branch_metrics[key] = bm
            del pmaps, gts
            res = {
                "paper": "ObjectCore (WACV 2026) - reimplementation",
                "category": args.category, "phase": phase, "k": k, "seed": seed,
                "support_images": [str(p.relative_to(Path(args.data_root))) for p in sup_paths],
                "auroc": aurocs(S, labels, kinds),
                "auroc_structural_branch_only": aurocs(SA, labels, kinds),
                "auroc_logical_branch_only": aurocs(SL, labels, kinds),
                "metrics": metrics,                    # paper Table 1 metrics (all test images)
                "metrics_branch_only": branch_metrics,  # paper Table 3 "Only MS" / "Only ML"
                "n_gt_missing": n_gt_missing,
                "maps_saved": save_maps,
                "objects_per_support": [len(r["objs"]) for r in sup_recs],
                "mean_objects_per_test": float(np.mean([r["n_objects"] for r in rows])),
                "n_test": len(rows),
                "runtime_s": round(time.time() - t0, 1),
                "args": vars(args),
                "implementation_choices": IMPLEMENTATION_CHOICES,
            }
            with open(run_dir / "scores.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(run_dir / "results.json", "w") as f:
                json.dump(res, f, indent=2)
            a, mt = res["auroc"], res["metrics"]
            log(f"  RESULT k={k} seed={seed}: I-AUROC all {a['all']:.2f} | logical {a['logical']:.2f} | "
                f"structural {a['structural']:.2f} | mean(L,S) {a['mean_LS']:.2f} | I-F1 {mt['I-F1-max']:.1f} | "
                f"P-AUROC {mt['P-AUROC']:.1f} | P-F1 {mt['P-F1-max']:.1f} | {res['runtime_s']:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
