# Implementation notes

## Code availability
- The paper points to github.com/MaticFuc/ObjectCore, but that repository is **not public** (checked 1 Oct 2026: 404, and `git ls-remote` asks for credentials).
- The authors' public repositories (SALAD, AnomalyVFM, TransFusion, an anomalib fork) do not contain ObjectCore.
- github.com/smithdak/objectcore is an unrelated TypeScript project that happens to share the name.
- So everything here is an **independent reimplementation from the paper and its supplementary material**. We plan to email the authors to ask for the code.

## Choices the paper does not specify
All of these are command-line flags of `objectcore.py` and are written to every `results.json` under `implementation_choices`.

| Item | Our choice | Flag |
|---|---|---|
| "First three layers" | Swin stage 1, 2, 3 outputs (strides 4 / 8 / 16; 128 / 256 / 512 channels), taken before patch merging | - |
| Combining layers | each stage gets the 5 × 5 mean filter at its own resolution, is L2-normalised per patch, resized to a common 100 × 100 grid, then the stages are concatenated (896-d) | `--feat_res 100` |
| Mask size unit | fraction of the image area | - |
| Padding cost (unequal object counts) | maximum cosine distance (2) + γ · 1 | - |
| Combining A_S and A_L | added **without rescaling** on the feature grid, upsampled to 256 × 256, then Gaussian σ = 4 | `--map_res 256` |
| NMS on detections | none (the paper mentions none) | `--nms` |
| Which detector parameters are fine-tuned | all | - |
| Weight decay / gradient clip | 1e-4 / 0.1 (GroundingDINO defaults) | `--ft_wd`, `--ft_clip` |
| Auxiliary losses during fine-tuning | decoder auxiliary + encoder two-stage classification losses on; box losses weighted 0 | - |
| Tokens in the classification loss | `all` = Hugging Face default (includes [CLS] / "." / [SEP]); `phrase` = our fix, "part" token only | `--ft_loss_tokens` |
| Detector reset | reloaded from pretrained weights for every support draw | - |
| I-AUROC | "all" (good vs all anomalies) is compared with the paper; logical, structural and mean(L, S) are also reported | - |
| Pixel metrics | P-AUROC and P-F1-max over all test pixels at 256 × 256; ground truth = union of all LOCO GT masks of an image | `--pixel_res 256` |
| Averaging | per draw, mean of the 5 categories; then mean ± population std over the 3 draws | - |
| Support draw | `random.Random(1000·k + seed)`, seeds 0, 1, 2; the drawn image names are saved in `results.json` | `--draw_seeds` |

## Hardware and runtime
- AWS (Monash RACE) g5.2xlarge: 1 × NVIDIA A10G 24 GB, us-east-1, Ubuntu 22.04, driver 550.144.03.
- Run 1 (both phases, 5 categories × k = 1, 2, 4 × 3 draws = 90 runs): about 5 h 10 min, about 6–7 USD.
- The paper used 1 × A100: about 5 min of fine-tuning plus inference per category, 215 ms per image.

## Folder layout on the workspace
`setup.sh` copies the code into `~/objectcore` (flat layout). Each `run_all.sh` launch creates a new folder `~/objectcore/runs/<YYYY-MM-DD_HHMM>_<mode>/` containing:
- `code/`: copy of the code plus SHA256SUMS
- `results/`
- `logs/`
- `run_record.txt`
- `report/`

`runs/latest` points to the newest run. Resume an interrupted run with `RESUME=<folder> bash run_all.sh <mode>`.

## Testing
- The full pipeline (both phases, scoring, metrics, report, the four step scripts) was smoke-tested on CPU with small random-weight GroundingDINO and SAM models.
- We checked that fine-tuning lowers the loss and leaves the box head unchanged (box loss weight 0).
- The real run used the real weights and the full MVTec LOCO.
