# Source code

`objectcore.py` is the complete ObjectCore reimplementation, the exact file used for run 1 on AWS
(plus the `--ft_loss_tokens` / `--ft_clip` flags, whose defaults reproduce run 1).

| Part | Where |
|---|---|
| Data loading (MVTec LOCO layout) | `list_images`, `load_category`, `load_rgb` |
| GroundingDINO detection, Swin layer 1-3 features, SAM masks | `class Models` (`detect`, `features`, `masks`) |
| Per-image description: features, objects (o, mask, size) | `describe_image` |
| Memory banks M_S / M_L, A_S, A_L, final score | `class ObjectCore` (`structural_map`, `logical_map`, `score`) |
| Detector fine-tuning (L_cls only) + phrase-token fix | `finetune_detector`, `phrase_token_loss` |
| Metrics | `aurocs`, `f1_max`, `load_gt`, `pixel_metrics` |
| Paper protocol for one category (k x draws, writes results.json / scores.jsonl / maps) | `main` |

Every choice the paper leaves open is in `IMPLEMENTATION_CHOICES` at the top of the file and is written into
every `results.json`. Run it directly:

```bash
python 02_src/source_code/objectcore.py --data_root ~/objectcore/data/mvtec_loco --category breakfast_box \
    --shots 1 2 4 --draws 3 --finetune --ft_loss_tokens phrase --out_dir results
```
