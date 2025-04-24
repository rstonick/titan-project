#!/bin/bash

#SBATCH --job-name=batch_1gpu_100k         # Job name
#SBATCH --gres=gpu:h200:1                # Request 1 H100 GPU (adjust type if needed)
#SBATCH -N1                              # Request 1 node
#SBATCH --ntasks-per-node=1              # Run one task per node
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=200G                        # Memory for H100 node (adjust if using H200)
#SBATCH --time=12:00:00                  # Time limit: 12 hours (buffer for ~8hr run)
#SBATCH --output=batch_1gpu_100k_%j.log  # Standard output and error log

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Requesting 1 GPU."

# --- Environment Setup ---
echo "Loading modules..."
module purge
module load anaconda3 cuda/12.1.1 # Verify modules
echo "Modules loaded."

# --- Set Environment Variables ---
export HF_DATASETS_CACHE="$HOME/scratch/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$HOME/scratch/.cache/huggingface/models"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export WANDB_DIR="$HOME/scratch/titan-project/wandb"
export WANDB_CACHE_DIR="$HOME/scratch/.cache/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Environment variables set."

# --- Activate Conda Environment ---
CONDA_ENV_PATH="$HOME/scratch/titan_env_vm"
echo "Activating Conda environment: $CONDA_ENV_PATH"
source activate "$CONDA_ENV_PATH"
echo "Conda environment activated."

# --- Navigate to Project Directory ---
PROJECT_DIR="$HOME/scratch/titan-project"
echo "Changing to directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1 # Exit if cd fails

# --- Run the Training Script ---
CONFIG_FILE="configs/config_batch_1gpu_100k.yaml" # Use the 1GPU config file
echo "Running training script with config: $CONFIG_FILE"

# Ensure TrainingArguments in the script reads all necessary parameters from config
python train_mistral_lora.py --config "$CONFIG_FILE"

EXIT_CODE=$?
echo "Training script finished with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE