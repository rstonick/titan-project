#!/bin/bash

#SBATCH --job-name=batch_4gpu_100k       # Job name
#SBATCH --gres=gpu:h200:4                # Request 4 H200 GPUs (adjust type if needed)
#SBATCH -N1                              # Request 1 node
#SBATCH --ntasks-per-node=4              # Number of tasks (match GPU count)
#SBATCH --cpus-per-task=8                # CPUs per task/GPU
#SBATCH --mem=600G                       # Memory for 4xH100 node (adjust based on node spec)
#SBATCH --time=04:00:00                  # Time limit: 4 hours (buffer for ~2.7hr run)
#SBATCH --output=batch_4gpu_100k_%j.log  # Standard output and error log

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Requesting 4 GPUs."

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
# Set by Accelerate, but good practice:
# export WORLD_SIZE="$SLURM_NTASKS" # Accelerate usually handles this
# export RANK="$SLURM_PROCID"       # Accelerate usually handles this
# export LOCAL_RANK="$SLURM_LOCALID" # Accelerate usually handles this
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

# --- Run the Training Script using Accelerate ---
CONFIG_FILE="configs/config_batch_4gpu_100k.yaml" # Use the 4GPU config file
NUM_GPUS=4 # Match --gres requested
echo "Running training script with accelerate on $NUM_GPUS GPUs using config: $CONFIG_FILE"

# Ensure TrainingArguments in the script reads all necessary parameters from config
# Run accelerate config once beforehand if needed
accelerate launch --multi_gpu --num_processes=$NUM_GPUS train_mistral_lora.py --config "$CONFIG_FILE"

EXIT_CODE=$?
echo "Training script finished with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE