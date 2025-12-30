#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

mkdir -p .torch_cache .mplconfig .cache/fontconfig
export TORCH_HOME="$ROOT_DIR/.torch_cache"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$ROOT_DIR/.mplconfig"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"

python3 - <<'PY'
try:
  import torch  # noqa: F401
  import torchvision  # noqa: F401
except Exception as e:
  raise SystemExit(
    "PyTorch/TorchVision not installed. Please install dependencies first:\n"
    "  python3 -m venv .venv\n"
    "  source .venv/bin/activate\n"
    "  pip install -r requirements.txt\n\n"
    f"Original error: {e}"
  )
print("OK: torch/torchvision import")
PY

if [[ ! -f "pretrained/mobilenetv2_meme_best.pt" ]]; then
  echo "Missing weights: pretrained/mobilenetv2_meme_best.pt"
  echo "Train first, e.g.: python3 train.py --data_dir data/demo --epochs 10 --batch_size 16 --lr 1e-4 --freeze_backbone"
  exit 1
fi

if [[ ! -d "data/demo/test" ]]; then
  echo "Missing demo split: data/demo/test"
  echo "Create it first, e.g.: python3 prepare_dataset.py --raw_dir data/demo/raw --out_dir data/demo --train 0.7 --val 0.15 --test 0.15"
  exit 1
fi

python3 eval.py \
  --data_dir data/demo \
  --weights pretrained/mobilenetv2_meme_best.pt \
  --out_dir outputs/eval_demo \
  --num_workers 0

echo "Done. See outputs/eval_demo/metrics.json and outputs/eval_demo/confusion_matrix.png"
