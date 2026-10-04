# Environment setup

## Hardware used
| Item | Value |
|---|---|
| Machine | AWS g5.2xlarge (Monash RACE workspace), us-east-1 |
| GPU | 1 × NVIDIA A10G 24 GB, driver 550.144.03 |
| OS | Ubuntu 22.04, 291 GB disk |
| Python | 3.10 (uv virtual environment at `~/objectcore/env`) |
| Libraries | torch 2.5.1 + CUDA 12.1, transformers 4.57.1, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.7.2 |

Any GPU with 24 GB or more should work: fine-tuning GroundingDINO Swin-B at 800 × 800 with batch 1 fits on the A10G.

## One-command setup (about 15 min)
```bash
git clone https://github.com/iamabroor/Few_Shot_ObjectCore.git
cd Few_Shot_ObjectCore
bash 03_scripts/setup.sh
```
`setup.sh` does the following:
1. Creates `~/objectcore/{data,runs,logs,hf_cache}` and copies the code there (flat layout).
2. Installs uv, creates the Python 3.10 environment, and installs PyTorch 2.5.1 (cu121) and `06_environment/requirements.txt`.
3. Downloads **MVTec LOCO AD** (5.7 GB, official MVTec link) to `~/objectcore/data/mvtec_loco` and prints the image counts.
4. Downloads the GroundingDINO Swin-B and SAM ViT-B weights to `~/objectcore/hf_cache`.
5. Writes `~/objectcore/logs/run_record.txt` (GPU, driver, library versions, code hash).

## Manual setup
```bash
uv venv ~/objectcore/env --python 3.10 && source ~/objectcore/env/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r 06_environment/requirements.txt
export HF_HOME=~/objectcore/hf_cache
```
Get MVTec LOCO from https://www.mvtec.com/company/research/datasets/mvtec-loco (licence CC BY-NC-SA 4.0). The expected layout is:
```
mvtec_loco/<category>/train/good/*.png
mvtec_loco/<category>/test/{good,logical_anomalies,structural_anomalies}/*.png
mvtec_loco/<category>/ground_truth/{logical,structural}_anomalies/<id>/*.png
```

## Running
```bash
cd ~/objectcore && tmux new -s oc
bash run_all.sh smoke        # ~5 min check
bash run_all.sh all          # full protocol, Phase A + Phase B (~5 h on an A10G)
EXTRA_ARGS="--ft_loss_tokens phrase" bash run_all.sh B   # Phase B with the phrase-token fix
# Ctrl+B then D to detach; tmux attach -t oc to come back
```

## Workspace notes
- Always activate the environment first: `source ~/objectcore/env/bin/activate && export HF_HOME=~/objectcore/hf_cache`.
- If the DCV desktop keeps opening the GNOME overview, run `gsettings set org.gnome.desktop.interface enable-hot-corners false`.
