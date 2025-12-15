# Mammo-CTIP: Containerized Evaluation on SCC

This repository provides a Singularity container for evaluating the Mammo-CTIP model on the BU SCC cluster.

## Environment Requirement

- Platform: BU SCC (Linux)
- Container runtime: Singularity
- GPU: NVIDIA GPU with at least 12 GB VRAM
- CUDA support required (use `--nv` flag)

## Pulling the Container

Pull the container image from GitHub Container Registry:

```bash
singularity pull docker://ghcr.io/oliver2087/mammo-ctip:main
```

This will generate the following image file:

```
mammo-ctip_main.sif
```

## Running the Default Evaluation Script

On SCC, run the following command to execute the default evaluation script inside the container:

```bash
singularity run --nv \
  -B /projectnb:/projectnb \
  mammo-ctip_main.sif \
  --run eval \
  --ckpt_path /projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/TABWFOCFullSize/last_epoch.pt \
  --csv_path /projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv \
  --img_root /projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/dataset/cohort_1_process \
  --eval_mode clip
```

This command runs the built-in evaluation pipeline in CLIP evaluation mode.

## Custom Input Configuration

To evaluate the model on custom data, modify the following arguments:

- `--csv_path`: Path to the input CSV file
- `--img_root`: Root directory of the corresponding image dataset

When using custom inputs, you must ensure:

1. The CSV file follows the same format and column structure as  
   `Embed_classification_diagnostic_data_wo_dup_folds_update.csv`
2. Image file paths and directory structure match the entries specified in the CSV
3. All data are accessible under the mounted directory (`/projectnb`)

For reference, please inspect the structure and content of  
`Embed_classification_diagnostic_data_wo_dup_folds_update.csv` at  
"/projectnb/ec500kb/projects/Fall_2025_Projects/Project_3/Embed_classification_diagnostic_data_wo_dup_folds_update.csv"

## Notes
- This container is intended to be run on BU SCC only
- No container modification is needed for evaluation
