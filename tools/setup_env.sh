#!/bin/bash
# Create the conda env for this cookbook and install all dependencies.
# Run from the repo root: bash tools/setup_env.sh
#
# Python 3.12 + the sagemaker-train SDK (from PyPI) + this package and its
# extras. Override the env name via ENV_NAME, or install the SDK from a local
# wheel by setting SAGEMAKER_TRAIN_WHEEL.
set -euo pipefail

ENV_NAME="${ENV_NAME:-cookbook}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# By default the SDK is installed from PyPI; set SAGEMAKER_TRAIN_WHEEL to a
# local wheel path to use a specific build instead.
WHEEL="${SAGEMAKER_TRAIN_WHEEL:-}"

if [ -n "$WHEEL" ] && [ ! -f "$WHEEL" ]; then
    echo "ERROR: sagemaker-train wheel not found at:"
    echo "    $WHEEL"
    echo "Unset SAGEMAKER_TRAIN_WHEEL to install from PyPI, or point it at a valid wheel."
    exit 1
fi

# --- Conda env ----------------------------------------------------------------
# Create the env if it does not already exist; otherwise install into it.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
    echo "Conda env '$ENV_NAME' already exists, installing into it."
fi

run() { conda run -n "$ENV_NAME" "$@"; }

# --- PySDK --------------------------------------------------------------------
# Install the training-session SDK. torch CPU is sufficient (training runs on
# the remote service, not locally), so pull torch from the CPU index.
if [ -n "$WHEEL" ]; then
    echo "Installing the sagemaker-train SDK from wheel..."
    run pip install --extra-index-url https://download.pytorch.org/whl/cpu "$WHEEL"
else
    echo "Installing the sagemaker-train SDK from PyPI..."
    run pip install --extra-index-url https://download.pytorch.org/whl/cpu --upgrade sagemaker-train
fi

echo "Pinning transformers==5.5.4 and numpy==2.5.0..."
run pip install "transformers==5.5.4" "numpy==2.5.0"

# --- This cookbook + extras ---------------------------------------------------
# plots  -> matplotlib (sft/analyze.py, tools/plot_metrics.py)
# code-rl -> aiohttp + datasets (--env_type code_rl)
# dev    -> pytest + hypothesis (the test suite)
echo "Installing the cookbook and extras..."
run pip install -e '.[plots,code-rl,dev]'

# --- Verify -------------------------------------------------------------------
echo "Verifying installation..."
run python -c "from sagemaker.train.training_session import ServiceClient, LossFunction, AdamParams, SamplingParams; print('PySDK OK')"
run python -c "import rl.train, sft.train, service; print('cookbook OK')"

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Run the tests with:   conda run -n $ENV_NAME python -m pytest tests/"
