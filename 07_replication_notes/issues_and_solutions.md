# Issues and solutions

## Research issues

### 1. Official code not public
- **Issue:** the paper's repository returns 404.
- **Solution:** full reimplementation from the paper. Every unstated choice is exposed as a flag and logged in `results.json`, so each run documents exactly what was assumed.

### 2. Detector collapse after fine-tuning (main finding)
- **Issue:** with the paper's fine-tuning recipe (100 epochs, lr 1e-5, L_cls only), the detector *loses* objects instead of getting better. Examples from the run 1 logs (k = 1, seed 2):

  | Category | Pseudo-detections before | Objects after fine-tuning |
  |---|---|---|
  | breakfast_box | 7 | 0 |
  | screw_bag | 2 | 0 |
  | splicing_connectors | 3 | 1 |
  | pushpins | 15 | 15 |

  The loss starts at 5,000–35,000 and ends at 80–700.
- **Cause:** the Hugging Face GroundingDINO loss is a sigmoid focal loss over **all** prompt tokens. Its text mask includes `[CLS]`, `.` and `[SEP]` (BERT ids 101, 1012, 102), and every one of the 900 queries is trained as a negative on them. The detection score uses those token probabilities, so after fine-tuning every box drops below the 0.25 threshold.
- **Diagnostic** (`03_scripts/diag_finetune.py`, breakfast_box `266.png`, 100 epochs; raw output on the workspace in `logs/diag_finetune.txt`):

  | Variant | Objects on support (target 7) | Objects on 3 test images |
  |---|---|---|
  | V1: paper setting (all tokens) | 1 | [1, 1, 1] |
  | **V2: phrase-token loss (our fix)** | **7** | **[6, 6, 6]** |

  V3 (phrase-token loss, no gradient clipping) also ran; its numbers are in the same log file.
- **Solution (our contribution):** `--ft_loss_tokens phrase` (`phrase_token_loss` in `objectcore.py`) restricts the classification loss to the phrase token ("part") by masking the special tokens out of `build_text_mask`. Config: `04_configs/phase_B_phrase_fix.yaml`.
- **Status:** the full Phase B rerun with the fix is the next experiment: `EXTRA_ARGS="--ft_loss_tokens phrase" bash run_all.sh B`. Its results will go into `05_results/`.

### 3. The logical branch hurts instead of helping
- **Issue:** at k = 4 (run 1, Phase B), the **structural-only** branch scores 70.3 I-AUROC, above the paper's "only M_S" figure of 68.7, and it beats the full model in **every** category. The **logical-only** branch scores 60.5, against 70.6 in the paper.
- **Causes found:**
  1. Detector collapse (issue 2): with 0–1 objects there is nothing to match.
  2. Object counts are noisy even on *normal* images: pushpins 2–17 objects per image, screw_bag 1–11. Count differences between normal images then look like logical anomalies.
  3. juice_bottle gets 1 object per image, so its logical branch degenerates to a whole-image comparison.
  4. The A_L scale (unmatched objects add their *whole mask* with value 1) is much larger than the A_S scale (a cosine distance that is usually small), so the logical map dominates the sum.
- **Mitigation tested:** rescaling the two maps before fusion (fusion check) gained 3–4 I-AUROC points but stayed below structural-only.
- **Next steps:** the phrase-token fix (more stable objects), optional NMS (`--nms 0.7`), per-branch normalisation computed on the support images, and asking the authors how A_S and A_L are scaled.

### 4. Results vs the paper
- Phase B (paper setting) I-AUROC is 60.8 / 66.3 / 63.5 at k = 1 / 2 / 4, against 72.5 / 74.6 / 80.8 in the paper.
- Phase A (no fine-tuning) is 57.8 / 58.8 / 61.4. The paper's ablation implies about 69.0 at k = 4.
- Pixel-level metrics are close to the paper, and P-F1 is above the paper at every k (see `report/tables/table1_comparison.csv`). Localisation therefore works; image-level logical reasoning is where the gap is.

## Engineering issues

| Issue | Solution |
|---|---|
| `run_all.sh` printed "ALL DONE" even when a Python step failed (the exit status was lost in the pipe to `tee`) | check `PIPESTATUS`, collect failures, and print `!!! FAILED` / `FINISHED WITH ERRORS` |
| `gaussian_blur` reflect padding failed on a 2-D tensor | pad the 4-D tensor with explicit `(r, r, 0, 0)` / `(0, 0, r, r)` |
| Small SAM test model: prompt encoder size mismatch | rebuilt the test model (affected the CPU smoke test only) |
| The v2 package on the workspace lacked `make_report.py` | re-ran `setup.sh` with v3; setup now always copies the full code |
| `python: command not found` on the workspace | activate the environment first: `source ~/objectcore/env/bin/activate` |
| DCV desktop kept opening the GNOME overview (hot corner) | `gsettings set org.gnome.desktop.interface enable-hot-corners false`, or work over SSH |
| Uploaded fix package not found on the workspace (`find /` empty) | re-sent the package and uploaded it again |
| Runs overwriting each other | every launch creates a new time-stamped run folder with a copy of the code and a run record; `RESUME=` continues a run in place |

## Points for a presentation slide
- The official code is private, so we built a full reimplementation from the paper with every hidden choice logged.
- We **found and fixed a detector-collapse bug**: the HF GroundingDINO loss trains `[CLS]`, `.` and `[SEP]` as negatives. Our phrase-token loss keeps 7 of 7 objects instead of 1 of 7.
- The structural branch **reproduces** the paper (70.3 vs 68.7), and pixel metrics match or beat it.
- The logical branch is weak because of the collapse, noisy object counts and unbalanced map fusion. Rescaling gives +3–4 points.
- Reproducibility: a new run folder per launch, a code hash, a run record, and fixed support-draw seeds.
