#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE ======
PYFILE="train_imgTabCLIP_Focal_wFocBCE_resample_full.py" 
SAVE_DIR="./runs/run1"
CSV_PATH="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv"
IMG_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/dataset/cohort_1_process"
NUM_EPOCHS=10

# Option A: per-class overrides (repeatable)
ALPHAS=(
  "cancer=0.75"
  "mass=0.25"
  "calc=0.25"
)

# Option B: full JSON override (leave empty to disable)
# NOTE: keep this as a single line.
CLASS_ALPHA_JSON=''  # e.g. '{"mass":0.2,"calc":0.25,"cancer":0.8}'

# ====== OPTIONAL: activate env (uncomment if needed) ======
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env
# or: micromamba activate your_env

mkdir -p "${SAVE_DIR}"

CMD=(python "${PYFILE}"
  --run train
  --save_dir "${SAVE_DIR}"
  --csv_path "${CSV_PATH}"
  --img_root "${IMG_ROOT}"
  --num_epochs "${NUM_EPOCHS}"
)

# add JSON if provided
if [[ -n "${CLASS_ALPHA_JSON}" ]]; then
  CMD+=( --class_alpha_config "${CLASS_ALPHA_JSON}" )
fi

# add per-key overrides
for a in "${ALPHAS[@]}"; do
  CMD+=( --alpha "${a}" )
done

echo "Running:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"