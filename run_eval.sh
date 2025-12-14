#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE ======
PYFILE="train_imgTabCLIP_Focal_wFocBCE_resample_full.py"   # <-- your script name
CKPT_PATH="./TABWFOCFullSize/last_epoch.pt"                     # or best_model.pt
CSV_PATH="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv"
IMG_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/dataset/cohort_1_process"
EVAL_MODE="case"   # case | clip | both (depending on your evaluate_model)

# ====== OPTIONAL: activate env (uncomment if needed) ======
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

python "${PYFILE}" \
  --run eval \
  --ckpt_path "${CKPT_PATH}" \
  --csv_path "${CSV_PATH}" \
  --img_root "${IMG_ROOT}" \
  --eval_mode "${EVAL_MODE}"