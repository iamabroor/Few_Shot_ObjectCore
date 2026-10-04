# Configurations

Each YAML key is a flag of `02_src/source_code/objectcore.py` (`key: value` -> `--key value`).
A config can start from another one with `base:`.

| File | Use |
|---|---|
| `objectcore_default.yaml` | All settings with their defaults; paper values are marked `paper` |
| `phase_A_zeroshot.yaml` | Phase A: pretrained detector, no fine-tuning (paper ablation "no fine-tuning") |
| `phase_B_finetune.yaml` | Phase B exactly as in the paper, with the Hugging Face default loss. This is run 1 |
| `phase_B_phrase_fix.yaml` | Phase B with our fix: classification loss only on the "part" token |
| `smoke_test.yaml` | ~5 min check: juice_bottle, 4-shot, 1 draw, 40 test images |

Used by `03_scripts/preprocessing.py` and `03_scripts/training.py`, for example
`python 03_scripts/training.py --config phase_B_phrase_fix.yaml --category pushpins --k 4 --seed 0 --out models/pp`.

`run_all.sh` (full protocol) uses the `objectcore.py` defaults, which are the same values as
`phase_A_zeroshot.yaml` / `phase_B_finetune.yaml`. To run the full protocol with the fix:
`EXTRA_ARGS="--ft_loss_tokens phrase" bash run_all.sh B`.
