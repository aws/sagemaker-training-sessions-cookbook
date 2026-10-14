#!/bin/bash
# Create the conda env for this cookbook and install all dependencies.
# Run from the repo root: bash tools/setup_env.sh
#
# Python 3.12 + the sagemaker-train SDK (from PyPI) + this package and its
# extras. Override the env name via ENV_NAME.
set -euo pipefail

ENV_NAME="${ENV_NAME:-cookbook}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# --- Conda env ----------------------------------------------------------------
# Create the env if it does not already exist; otherwise install into it.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
    echo "Conda env '$ENV_NAME' already exists, installing into it."
fi

run() { conda run -n "$ENV_NAME" "$@"; }

# --- SDK ----------------------------------------------------------------------
echo "Installing the sagemaker-train SDK from PyPI..."
run pip install "sagemaker-train>=1.25.0"

# --- This cookbook + extras ---------------------------------------------------
# plots  -> matplotlib (sft/analyze.py, tools/plot_metrics.py)
# code-rl -> aiohttp + datasets (--env_type code_rl)
# dev    -> pytest + hypothesis (the test suite)
echo "Installing the cookbook and extras..."
run pip install -e '.[plots,code-rl,dev]'

# --- Verify -------------------------------------------------------------------
echo "Verifying installation..."
run python -c "from sagemaker.train.training_session import ServiceClient, LossFunction, AdamParams, SamplingParams; print('SDK OK')"
run python -c "import training_sessions_cookbook.rl.train, training_sessions_cookbook.sft.train, training_sessions_cookbook.preference.train, training_sessions_cookbook.service; print('cookbook OK')"

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Run the tests with:   conda run -n $ENV_NAME python -m pytest tests/"
