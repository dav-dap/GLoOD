#!/usr/bin/env bash
set -euo pipefail
###############################################################################
# build_env_base.sh - base Conda env without PyTorch
###############################################################################

PYTHON_VER="3.11"
ENV_NAME="env_base"

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
  typer==0.16.0
  pyyaml==6.0.2
  pyarrow==21.0.0
  pyvista==0.48.0
  vtk==9.6.1
  git+https://gitlab.com/hslur/hslur.git@9710418ea8f1dfc3340cf30c9e1fb0a2c7445b64
)

echo -e "\n[1/4]  Create env ..."
conda create -y -n "$ENV_NAME" "python=$PYTHON_VER" pip

echo -e "\n[2/4]  Activate env ..."
source activate "$ENV_NAME"

echo -e "\n[3/4]  Extras ..."
python -m pip install --no-cache-dir "${EXTRAS[@]}"

echo -e "\n[4/4]  Import checks ..."
python3 - <<'PY'
import einops
import hydra
import matplotlib
import openpyxl
import pandas
import pyarrow
import pyvista
import sklearn
import tabulate
import tqdm
import typer
import vtk
import yaml

print("Base environment imports OK")
PY

echo -e "\nFinished - activate anytime with:  source activate $ENV_NAME"
