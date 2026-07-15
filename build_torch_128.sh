#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# build_tf2.19_gpu_conda.sh  –  TF 2.19.* (hermetic CUDA 12) + extras
###############################################################################

CUDA_VER="128"
PYTHON_VER="3.11"
ENV_NAME="env-torch-"$CUDA_VER

EXTRAS=(
  hydra-core==1.3.2
  hydra-submitit-launcher==1.2.0
  matplotlib==3.10.3
  openpyxl==3.1.5
  pandas==2.3.0
  tabulate==0.9.0
  tqdm==4.67.1
  scikit-learn==1.7.0
  einops==0.8.1
  timm==0.6.13
  typer==0.16.0
  pyyaml==6.0.2
  pyarrow==21.0.0
  pyvista==0.48.0
  vtk==9.6.1
  git+https://gitlab.com/hslur/hslur.git@9710418ea8f1dfc3340cf30c9e1fb0a2c7445b64
)

echo -e "\n[1/5]  Create env ..."
conda create -y -n "$ENV_NAME" "python=$PYTHON_VER" pip

echo -e "\n[2/5]  Activate env ..."
source activate "$ENV_NAME"

echo -e "\n[3/5]  PyTorch GPU (hermetic wheel) ..."
python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu$CUDA_VER

echo -e "\n[4/5]  Extras ..."
python -m pip install --no-cache-dir "${EXTRAS[@]}"

echo -e "\n[5/5]  Tests ..."
python3 - <<'PY'
import torch
x = torch.rand(5, 3)
print(x)
PY

python3 - <<'PY'
import torch
print(f'Cuda availability on this system: {torch.cuda.is_available()}')
PY

echo -e "\n✅  Finished — activate anytime with:  source activate $ENV_NAME"
