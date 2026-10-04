#!/usr/bin/env python3
"""
ObjectCore - step-by-step visualisation of ONE test image (every pipeline stage as its own image + a summary).

    cd ~/objectcore && source env/bin/activate && export HF_HOME=~/objectcore/hf_cache
    python viz_steps.py --category breakfast_box --image logical_anomalies/000.png
    python viz_steps.py --category pushpins --image logical_anomalies/010.png --finetune   # with detector fine-tuning

The support images are taken from a finished run (results.json of <run>/results/<phase>/<category>/k<k>_seed<seed>),
so the pictures match the numbers in the report. ObjectCore does not store a trained model: memory banks are rebuilt
from the support images (seconds); with --finetune the detector is fine-tuned again on them (~1-2 min).

Output: ~/objectcore/viz_steps/<category>_<image>/
  00_support_images.png      support set (k normal images)
  01_rgb.png                 test image
  02_groundingdino.png       detections, prompt "part" (support + test)
  03_sam_masks.png           SAM mask per detected box
  04_features.png            patch feature embeddings (Swin layers 1-3, PCA -> RGB)
  05_object_representations.png  each object filled with the colour of its representation vector (PCA -> RGB)
  06_memory_banks.png        structural bank (all support patches) + logical bank (object set of nearest support image)
  07_matching.png            Hungarian matching test <-> nearest support image (lines coloured by cost; red = unmatched)
  08_structural_map.png      A_S  nearest-neighbour patch distance
  09_logical_map.png         A_L  object-matching cost map
  10_final_anomaly.png       A = Gauss(A_S + A_L), score, ground truth, Normal / Anomaly decision
  summary.png                all steps in one figure
"""
import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))  # flat copy in ~/objectcore
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_src" / "source_code"))  # repo layout
import objectcore as oc  # noqa: E402

HOME = Path.home() / "objectcore"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--image", required=True, help="test image relative to <category>/test, e.g. logical_anomalies/000.png")
    ap.add_argument("--data_root", default=str(HOME / "data/mvtec_loco"))
    ap.add_argument("--run", default=str(HOME / "runs/latest"), help="run folder whose support draw is reused")
    ap.add_argument("--phase", default=None, help="A_zeroshot or B_finetune (default: B_finetune if --finetune else A_zeroshot)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune", action="store_true", help="fine-tune the detector on the support set first")
    ap.add_argument("--ft_loss_tokens", default="phrase", choices=["all", "phrase"])
    ap.add_argument("--out", default=str(HOME / "viz_steps"))
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--img_size", type=int, default=800)
    ap.add_argument("--feat_res", type=int, default=100)
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--ft_epochs", type=int, default=100)
    return ap.parse_args()


# ------------------------------------------------------------------------------------ helpers
def pca_rgb(feats_list):
    """feats_list: list of C x h x w tensors -> list of h x w x 3 images, one shared PCA."""
    X = torch.cat([f.float().flatten(1).T for f in feats_list], 0)
    X = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    out = []
    P = X @ V[:, :3]
    lo, hi = torch.quantile(P, 0.01, dim=0), torch.quantile(P, 0.99, dim=0)
    i = 0
    for f in feats_list:
        n = f.shape[1] * f.shape[2]
        p = ((P[i:i + n] - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        out.append(p.view(f.shape[1], f.shape[2], 3).numpy())
        i += n
    return out, V[:, :3], X.mean(0) if False else None


def obj_colors(objs_list):
    X = torch.cat([o for o in objs_list if len(o)], 0).float()
    if len(X) < 3:
        return [np.tile([[1, .5, 0]], (len(o), 1)) for o in objs_list]
    Xc = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(Xc, q=3, center=False)
    P = Xc @ V[:, :3]
    P = (P - P.min(0).values) / (P.max(0).values - P.min(0).values + 1e-8)
    P = 0.15 + 0.85 * P
    out, i = [], 0
    for o in objs_list:
        out.append(P[i:i + len(o)].numpy()); i += len(o)
    return out


def up(a, size):
    """h x w array -> image size (W, H)."""
    return np.array(Image.fromarray(np.asarray(a, dtype=np.float32)).resize(size, Image.BILINEAR))


def grid_masks_full(rec, size):
    return [up(m.float().numpy(), size) > 0.5 for m in rec["masks"]]


def centroid(m):
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean()) if len(xs) else (0, 0)


def gt_mask(data_root, cat, rel, size):
    kind = rel.split("/")[0]
    W, H = size
    if kind == "good":
        return np.zeros((H, W), bool)
    d = Path(data_root) / cat / "ground_truth" / kind / Path(rel).stem
    m = np.zeros((H, W), bool)
    for f in sorted(d.glob("*.png")) if d.exists() else []:
        m |= np.array(Image.open(f).convert("L").resize((W, H), Image.NEAREST)) > 0
    return m


def draw_boxes(ax, img, boxes, color="#00e5ff", title=None):
    ax.imshow(img); ax.set_axis_off()
    W, H = img.size
    for b in boxes:
        ax.add_patch(mpatches.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H,
                                        fill=False, lw=1.5, color=color))
    if title:
        ax.set_title(title, fontsize=9)


def paint(ax, img, masks, colors, title=None, alpha=0.6):
    ax.imshow(img); ax.set_axis_off()
    if masks:
        H, W = masks[0].shape
        canvas = np.zeros((H, W, 4))
        for m, c in zip(masks, colors):
            canvas[m, :3] = c[:3]; canvas[m, 3] = alpha
        ax.imshow(canvas)
    if title:
        ax.set_title(title, fontsize=9)


def heat(ax, img, amap, title, vmin=None, vmax=None, cmap="jet"):
    ax.imshow(img); ax.set_axis_off()
    vmin = np.percentile(amap, 1) if vmin is None else vmin
    vmax = np.percentile(amap, 99.9) if vmax is None else vmax
    im = ax.imshow(np.clip((amap - vmin) / max(vmax - vmin, 1e-8), 0, 1), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    return im


def save_single(fn, path, figsize):
    fig = plt.figure(figsize=figsize)
    fn(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  saved", path.name)


def f1_threshold(scores_file):
    if not scores_file.exists():
        return None
    rows = [json.loads(l) for l in open(scores_file)]
    s = np.array([r["score"] for r in rows]); y = np.array([r["label"] for r in rows]).astype(bool)
    best, thr = -1, None
    for t in np.unique(s):
        p = s >= t
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, thr = f1, t
    return float(thr)


# --------------------------------------------------------------------------------------- main
def main():
    a = get_args()
    phase = a.phase or ("B_finetune" if a.finetune else "A_zeroshot")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cat_root = Path(a.data_root) / a.category
    test_path = cat_root / "test" / a.image
    assert test_path.exists(), f"test image not found: {test_path}"
    out = Path(a.out) / f"{a.category}_{a.image.replace('/', '_').rsplit('.', 1)[0]}"
    out.mkdir(parents=True, exist_ok=True)

    # support set: same draw as in the run (fallback: same random draw rule as objectcore.py)
    run_dir = Path(a.run).resolve() / "results" / phase / a.category / f"k{a.k}_seed{a.seed}"
    if (run_dir / "results.json").exists():
        sup_rel = json.loads((run_dir / "results.json").read_text())["support_images"]
        sup_paths = [Path(a.data_root) / r for r in sup_rel]
        print(f"support set from {run_dir}")
    else:
        train = oc.list_images(cat_root / "train" / "good")
        sup_paths = random.Random(1000 * a.k + a.seed).sample(train, a.k)
        print(f"run results not found ({run_dir}); using the same random draw rule (k={a.k}, seed={a.seed})")
    thr = f1_threshold(run_dir / "scores.jsonl")

    args = types.SimpleNamespace(img_size=a.img_size, feat_res=a.feat_res, mean_kernel=5, box_threshold=a.box_threshold,
                                 text_threshold=a.text_threshold, nms=None, gamma=0.1, sigma=4.0, top_frac=0.01,
                                 map_res=256, ft_lr=1e-5, ft_wd=1e-4, ft_epochs=a.ft_epochs, ft_clip=0.1,
                                 ft_loss_tokens=a.ft_loss_tokens)
    print("loading GroundingDINO + SAM ...")
    M = oc.Models(a.gdino, a.sam, dev)
    sup_imgs = [oc.load_rgb(p) for p in sup_paths]
    test_img = oc.load_rgb(test_path)
    S = test_img.size

    if a.finetune:
        pseudo = [M.detect(im, args.img_size, args.box_threshold, args.text_threshold)[0] for im in sup_imgs]
        print(f"fine-tuning detector ({args.ft_epochs} epochs, loss tokens: {args.ft_loss_tokens}) ...")
        oc.finetune_detector(M, sup_imgs, pseudo, args, lambda s: print("   ", s.strip()))

    print("describing support + test images ...")
    sup_recs = [oc.describe_image(M, im, args) for im in sup_imgs]
    rec = oc.describe_image(M, test_img, args)
    core = oc.ObjectCore(sup_recs, args, dev)
    score, s_struct, s_logic, nn_idx, maps = core.score(rec)
    s_rec, s_img = sup_recs[nn_idx], sup_imgs[nn_idx]

    # shared visual encodings
    feat_rgb, _, _ = pca_rgb([rec["feat"]] + [r["feat"] for r in sup_recs])
    ocols = obj_colors([rec["objs"]] + [r["objs"] for r in sup_recs])
    t_masks = grid_masks_full(rec, S)
    sup_masks = [grid_masks_full(r, im.size) for r, im in zip(sup_recs, sup_imgs)]
    AS, AL, A = [up(maps[k].cpu().numpy(), S) for k in ["AS", "AL", "A"]]
    gt = gt_mask(a.data_root, a.category, a.image, S)
    kind = a.image.split("/")[0]
    verdict = None if thr is None else ("ANOMALY" if score >= thr else "NORMAL")

    # matching (same cost as objectcore.py)
    g = args.gamma
    ot, os_ = rec["objs"].float(), s_rec["objs"].float()
    st, ss = rec["sizes"], s_rec["sizes"]
    nt, ns = len(ot), len(os_)
    pairs, un_t, un_s = [], list(range(nt)), list(range(ns))
    if nt and ns:
        n = max(nt, ns)
        C = torch.full((n, n), 2.0 + g)
        C[:nt, :ns] = (1 - ot @ os_.T) + g * (st[:, None] - ss[None, :]).abs()
        r_, c_ = linear_sum_assignment(C.numpy())
        pairs = [(i, j, float(C[i, j])) for i, j in zip(r_, c_) if i < nt and j < ns]
        un_t = [i for i in range(nt) if i not in [p[0] for p in pairs]]
        un_s = [j for j in range(ns) if j not in [p[1] for p in pairs]]

    k = len(sup_imgs)
    tag = f"{a.category} | {a.image} | k={a.k}, seed {a.seed} | {'fine-tuned' if a.finetune else 'no fine-tuning'}"
    print(f"score {score:.4f} (structural {s_struct:.4f}, logical {s_logic:.4f}); threshold {thr}; objects test {nt}, nearest support {ns}")

    # ---------------------------------------------------------------- individual step images
    def f00(fig):
        for i, im in enumerate(sup_imgs):
            ax = fig.add_subplot(1, k, i + 1); ax.imshow(im); ax.set_axis_off()
            ax.set_title(f"support {i + 1}: {sup_paths[i].name}", fontsize=9)
        fig.suptitle(f"Step 0 – Few normal support images (k={k})")
    save_single(f00, out / "00_support_images.png", (3.2 * k, 3.2))

    def f01(fig):
        ax = fig.add_subplot(1, 1, 1); ax.imshow(test_img); ax.set_axis_off()
        ax.set_title(f"Step 1 – Test image ({kind})")
    save_single(f01, out / "01_rgb.png", (6, 5))

    def f02(fig):
        ax = fig.add_subplot(1, 2, 1)
        draw_boxes(ax, test_img, rec["boxes_norm"], title=f"test: {rec['n_boxes']} detections")
        ax = fig.add_subplot(1, 2, 2)
        draw_boxes(ax, s_img, s_rec["boxes_norm"], title=f"nearest support: {s_rec['n_boxes']} detections")
        fig.suptitle(f"Step 2 – GroundingDINO{' (fine-tuned)' if a.finetune else ''}, prompt \"part\", threshold 0.25")
    save_single(f02, out / "02_groundingdino.png", (11, 5))

    def f03(fig):
        ax = fig.add_subplot(1, 2, 1)
        cols = plt.get_cmap("tab20")(np.arange(max(nt, 1)) % 20)
        paint(ax, test_img, t_masks, cols, title=f"test: {nt} masks")
        ax = fig.add_subplot(1, 2, 2)
        cols = plt.get_cmap("tab20")(np.arange(max(ns, 1)) % 20)
        paint(ax, s_img, sup_masks[nn_idx], cols, title=f"nearest support: {ns} masks")
        fig.suptitle("Step 3 – SAM ViT-B masks (one per detected box)")
    save_single(f03, out / "03_sam_masks.png", (11, 5))

    def f04(fig):
        ax = fig.add_subplot(1, 2, 1); ax.imshow(test_img); ax.set_axis_off(); ax.set_title("test image", fontsize=9)
        ax = fig.add_subplot(1, 2, 2); ax.imshow(up(feat_rgb[0][..., 0], S)[..., None] * 0 + np.stack(
            [up(feat_rgb[0][..., c], S) for c in range(3)], -1)); ax.set_axis_off()
        ax.set_title("patch features (first 3 PCA components as RGB)", fontsize=9)
        fig.suptitle("Step 4a – Feature embeddings: Swin layers 1-3 of GroundingDINO, 5×5 mean filter")
    save_single(f04, out / "04_features.png", (11, 5))

    def f05(fig):
        ax = fig.add_subplot(1, 2, 1)
        paint(ax, test_img, t_masks, ocols[0], title="test objects", alpha=0.75)
        ax = fig.add_subplot(1, 2, 2)
        paint(ax, s_img, sup_masks[nn_idx], ocols[1 + nn_idx], title="nearest support objects", alpha=0.75)
        fig.suptitle("Step 4b – Object representations o = mean feature inside mask (similar colour = similar object)")
    save_single(f05, out / "05_object_representations.png", (11, 5))

    def f06(fig):
        for i, im in enumerate(sup_imgs):
            ax = fig.add_subplot(2, k, i + 1)
            ax.imshow(np.stack([up(feat_rgb[1 + i][..., c], im.size) for c in range(3)], -1)); ax.set_axis_off()
            ax.set_title(f"M_S: support {i + 1} patches", fontsize=8)
            ax = fig.add_subplot(2, k, k + i + 1)
            paint(ax, im, sup_masks[i], ocols[1 + i], alpha=0.75,
                  title=f"M_L: {len(sup_recs[i]['objs'])} objects" + ("  ← nearest" if i == nn_idx else ""))
        n_patch = sum(r["feat"].shape[1] * r["feat"].shape[2] for r in sup_recs)
        fig.suptitle(f"Step 5 – Memory banks: structural M_S = {n_patch:,} patch features (no coreset) | "
                     f"logical M_L = object set per support image")
    save_single(f06, out / "06_memory_banks.png", (3.2 * k, 6.6))

    def f07(fig):
        W1, H1 = S
        s_resized = s_img.resize(S)
        canvas = np.concatenate([np.asarray(test_img), np.asarray(s_resized)], 1)
        ax = fig.add_subplot(1, 1, 1); ax.imshow(canvas); ax.set_axis_off()
        sm = [m for m in grid_masks_full(s_rec, S)]
        cmap = plt.get_cmap("RdYlGn_r")
        for i, j, c in pairs:
            (x1, y1), (x2, y2) = centroid(t_masks[i]), centroid(sm[j])
            ax.plot([x1, x2 + W1], [y1, y2], color=cmap(min(c / 1.0, 1)), lw=2)
            ax.plot([x1], [y1], "o", color="white", ms=4); ax.plot([x2 + W1], [y2], "o", color="white", ms=4)
        for i in un_t:
            x, y = centroid(t_masks[i]); ax.plot([x], [y], "X", color="red", ms=12)
        for j in un_s:
            x, y = centroid(sm[j]); ax.plot([x + W1], [y], "X", color="red", ms=12)
        ax.set_title(f"left: test ({nt} objects) | right: nearest support ({ns} objects) | "
                     f"{len(pairs)} matched, {len(un_t)} extra, {len(un_s)} missing (red ×)", fontsize=9)
        sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); fig.colorbar(sm_, ax=ax, fraction=0.025, label="matching cost")
        fig.suptitle("Step 6 – Hungarian object matching, cost = cos-dist + 0.1·|size difference|")
    save_single(f07, out / "07_matching.png", (13, 5.5))

    def f08(fig):
        ax = fig.add_subplot(1, 1, 1); im = heat(ax, test_img, AS, f"A_S, structural-only score {s_struct:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("Step 7a – Structural anomaly map: nearest-neighbour patch distance to M_S")
    save_single(f08, out / "08_structural_map.png", (6.5, 5.5))

    def f09(fig):
        ax = fig.add_subplot(1, 1, 1); im = heat(ax, test_img, AL, f"A_L, logical-only score {s_logic:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle("Step 7b – Logical anomaly map: matched cost + unmatched (extra / missing) objects")
    save_single(f09, out / "09_logical_map.png", (6.5, 5.5))

    def f10(fig):
        ax = fig.add_subplot(1, 2, 1); heat(ax, test_img, A, f"A = Gauss(A_S + A_L), score {score:.3f}")
        ax = fig.add_subplot(1, 2, 2); ax.imshow(test_img); ax.set_axis_off()
        ax.imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.6)
        ax.set_title(f"ground truth ({kind})", fontsize=9)
        v = f"Decision: {verdict} (threshold {thr:.3f}, F1-optimal on the run's test scores)" if verdict else \
            "Decision: no threshold available (run scores not found)"
        fig.suptitle(f"Step 8 – Final anomaly map and score | {v}")
    save_single(f10, out / "10_final_anomaly.png", (11, 5))

    # --------------------------------------------------------------------------- summary
    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    axs = axs.ravel()
    axs[0].imshow(test_img); axs[0].set_axis_off(); axs[0].set_title(f"1. Test image ({kind})", fontsize=10)
    draw_boxes(axs[1], test_img, rec["boxes_norm"], title=f"2. GroundingDINO: {rec['n_boxes']} parts")
    paint(axs[2], test_img, t_masks, plt.get_cmap("tab20")(np.arange(max(nt, 1)) % 20), title=f"3. SAM masks ({nt})")
    axs[3].imshow(np.stack([up(feat_rgb[0][..., c], S) for c in range(3)], -1)); axs[3].set_axis_off()
    axs[3].set_title("4a. Feature embeddings (PCA)", fontsize=10)
    paint(axs[4], test_img, t_masks, ocols[0], title="4b. Object representations", alpha=0.75)
    paint(axs[5], s_img, sup_masks[nn_idx], ocols[1 + nn_idx], alpha=0.75,
          title=f"5. Logical bank: nearest support ({ns} obj.)")
    s_resized = s_img.resize(S)
    axs[6].imshow(np.concatenate([np.asarray(test_img), np.asarray(s_resized)], 1)); axs[6].set_axis_off()
    sm = grid_masks_full(s_rec, S)
    for i, j, c in pairs:
        (x1, y1), (x2, y2) = centroid(t_masks[i]), centroid(sm[j])
        axs[6].plot([x1, x2 + S[0]], [y1, y2], color=plt.get_cmap("RdYlGn_r")(min(c, 1)), lw=1.5)
    for i in un_t:
        x, y = centroid(t_masks[i]); axs[6].plot([x], [y], "X", color="red", ms=9)
    for j in un_s:
        x, y = centroid(sm[j]); axs[6].plot([x + S[0]], [y], "X", color="red", ms=9)
    axs[6].set_title(f"6. Hungarian matching ({len(un_t)} extra, {len(un_s)} missing)", fontsize=10)
    heat(axs[7], test_img, AS, f"7a. Structural map A_S ({s_struct:.3f})")
    heat(axs[8], test_img, AL, f"7b. Logical map A_L ({s_logic:.3f})")
    heat(axs[9], test_img, A, f"8. Final A, score {score:.3f}" + (f" → {verdict}" if verdict else ""))
    if gt.any():
        axs[9].contour(gt, levels=[0.5], colors="white", linewidths=1.2)
    fig.suptitle(f"ObjectCore pipeline – {tag}" + ("   (white contour = ground truth)" if gt.any() else ""), fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  saved summary.png")
    print(f"\nAll images in {out}")


if __name__ == "__main__":
    main()
