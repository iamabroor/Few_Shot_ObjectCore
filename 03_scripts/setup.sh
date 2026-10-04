#!/usr/bin/env bash
# ObjectCore replication - one-time setup on a fresh RACE workspace (Ubuntu 22.04, 1x A10G).
# Usage:  bash setup.sh
# EVERYTHING for ObjectCore lives in ONE folder, ~/objectcore:
#   env/       Python environment          data/      MVTec LOCO
#   hf_cache/  GroundingDINO + SAM weights runs/      one new sub-folder per launch of run_all.sh
#   *.py *.sh  the code
set -euo pipefail
ROOT=~/objectcore
SRC=$(cd "$(dirname "$0")" && pwd)          # .../Few_Shot_ObjectCore/03_scripts
REPO=$(cd "$SRC/.." && pwd)                  # repository root
mkdir -p "$ROOT"/{data,runs,logs,hf_cache}
export HF_HOME="$ROOT/hf_cache"
# always copy the latest code (overwrites older copies in ~/objectcore); the workspace copy is flat
cp -f "$REPO/02_src/source_code/objectcore.py" "$SRC"/*.py "$SRC"/*.sh "$ROOT"/
cp -rf "$REPO/04_configs" "$ROOT"/
cp -f "$REPO/README.md" "$ROOT"/ 2>/dev/null || true
cd "$ROOT"

echo "== 1/4 Python environment (uv, Python 3.10, PyTorch 2.5.1 + CUDA 12.1)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d "$ROOT/env" ] || uv venv "$ROOT/env" --python 3.10
source "$ROOT/env/bin/activate"
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r "$REPO/06_environment/requirements.txt"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "== 2/4 MVTec LOCO (5.7 GB, official MVTec link)"
if [ ! -d data/mvtec_loco/breakfast_box ]; then
  wget -q --show-progress -O data/mvtec_loco.tar.xz \
    "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz"
  mkdir -p data/mvtec_loco && tar -xf data/mvtec_loco.tar.xz -C data/mvtec_loco && rm data/mvtec_loco.tar.xz
fi
for c in breakfast_box juice_bottle pushpins screw_bag splicing_connectors; do
  printf "%-20s train %4s | test good %3s logical %3s structural %3s\n" $c \
    $(ls data/mvtec_loco/$c/train/good | wc -l) $(ls data/mvtec_loco/$c/test/good | wc -l) \
    $(ls data/mvtec_loco/$c/test/logical_anomalies | wc -l) $(ls data/mvtec_loco/$c/test/structural_anomalies | wc -l)
done

echo "== 3/4 Weights: GroundingDINO Swin-B + SAM ViT-B (Hugging Face)"
python - <<'EOF'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
for mid in ["IDEA-Research/grounding-dino-base"]:
    AutoProcessor.from_pretrained(mid); m = AutoModelForZeroShotObjectDetection.from_pretrained(mid)
    print(mid, "backbone:", m.config.backbone_config.model_type, "embed_dim", m.config.backbone_config.embed_dim,
          "depths", m.config.backbone_config.depths)
SamProcessor.from_pretrained("facebook/sam-vit-base"); SamModel.from_pretrained("facebook/sam-vit-base")
print("weights cached")
EOF

echo "== 4/4 Run record"
{
  date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  python -c "import torch,transformers,numpy,scipy,sklearn;print('torch',torch.__version__,'cuda',torch.version.cuda,'| transformers',transformers.__version__,'| numpy',numpy.__version__,'| scipy',scipy.__version__,'| sklearn',sklearn.__version__)"
  sha256sum objectcore.py
} | tee logs/run_record.txt
echo "Setup done. Next: bash run_all.sh smoke"
