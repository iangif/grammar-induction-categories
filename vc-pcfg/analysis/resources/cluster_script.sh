#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --account=def-eporte2
#SBATCH --job-name=en_smoke
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Load exactly the Python module used to create the environment.
module purge
module load python/3.10
module load StdEnv/2023
module load gcc arrow/21.0.0


source "$HOME/venvs/graminduct/bin/activate"

# sbatch should be called from the repository root.
REPO_ROOT="$HOME/grammar-induction-categories"
VCPCFG_DIR="${REPO_ROOT}/vc-pcfg"
DATA_PATH="${REPO_ROOT}/preprocessed-data/abstractscenes"
RUNS_DIR="${REPO_ROOT}/runs"

mkdir -p "$RUNS_DIR"

NUM_EPOCHS="${NUM_EPOCHS:-5}"
T_STATES=40
SEED="${SEED:-29}"
ENCODER_FILE="all_as-resn-50.npy"
IMG_DIM=2048
LOGGER="${RUNS_DIR}/en_smoke_joint_s${SEED}_e${NUM_EPOCHS}_c${T_STATES}"

echo "============================================================"
echo "Start time:    $(date)"
echo "Host:          $(hostname)"
echo "Repository:    $REPO_ROOT"
echo "Data:          $DATA_PATH"
echo "Output:        $LOGGER"
echo "Epochs:        $NUM_EPOCHS"
echo "Seed:          $SEED"
echo "Python:        $(which python)"
echo "============================================================"

# Fail early if important inputs are absent.
test -f "${VCPCFG_DIR}/as_train.py" || {
    echo "ERROR: as_train.py not found in ${VCPCFG_DIR}" >&2
    exit 1
}

test -d "$DATA_PATH" || {
    echo "ERROR: data directory not found: ${DATA_PATH}" >&2
    exit 1
}

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible to PyTorch")

print("GPU:", torch.cuda.get_device_name(0))
PY

cd "$VCPCFG_DIR"

python -u ./as_train.py \
    --num_epochs "$NUM_EPOCHS" \
    --encoder_file "$ENCODER_FILE" \
    --img_dim "$IMG_DIM" \
    --visual_mode \
    --logger_name "$LOGGER" \
    --seed "$SEED" \
    --data_path "$DATA_PATH" \
    --log_step 1000 \
    --vse_mt_alpha 1.0 \
    --vse_lm_alpha 1.0 \
    --t_states "$T_STATES"

echo "End time: $(date)"
echo "DONE"