#!/bin/bash

#-----------------------------------------------------------------------
# SLURM Directives
#-----------------------------------------------------------------------
#SBATCH --job-name=mistral_triviaqa_h100   # Job name for identification
#SBATCH --nodes=1                          # Request 1 node
#SBATCH --ntasks-per-node=1                # Run 1 task per node
#SBATCH --gres=gpu:h100:1                  # Request 1 H100 GPU *** This is the recommended starting point ***
#SBATCH --cpus-per-task=16                 # Request 16 CPU cores
#SBATCH --mem=150G                         # Request 150GB of system RAM
#SBATCH -t 08:00:00                        # Request 8 hours walltime (max allowed, adjust lower if confident)

# Define output and error log file paths
# IMPORTANT: Create this directory before submitting: mkdir -p ~/scratch/titan-project/slurm_logs
# %j will be replaced by the Slurm Job ID
#SBATCH --output=slurm_logs/mistral_%j.out   # Standard output log
#SBATCH --error=slurm_logs/mistral_%j.err    # Standard error log

#-----------------------------------------------------------------------
# Environment Setup
#-----------------------------------------------------------------------
echo "Setting up job environment..."
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

module purge                             # Clear existing modules
module load anaconda3 cuda/12.1.1        # Load Anaconda and CUDA

# Activate your specific conda environment (ensure path is correct)
source activate ~/scratch/titan_env_vm
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

# Set environment variables for Hugging Face/W&B caches in scratch
# Create directories if they don't exist
CACHE_DIR=~/scratch/.cache/huggingface
WANDB_CACHE_DIR=~/scratch/wandb
mkdir -p $CACHE_DIR/datasets
mkdir -p $CACHE_DIR/hub
mkdir -p $WANDB_CACHE_DIR

export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/hub
export HF_HOME=$CACHE_DIR
export WANDB_DIR=$WANDB_CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # OOM workaround

echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "WANDB_DIR: $WANDB_DIR"

#-----------------------------------------------------------------------
# Run the Training Script
#-----------------------------------------------------------------------
echo "Navigating to project directory..."
# Ensure you are in the directory containing your script and config
cd ~/scratch/titan-project
echo "Current directory: $(pwd)"

echo "Starting Python training script..."
# *** Ensure config.yaml is updated BEFORE submitting (see Step 2 below) ***
python train_mistral_lora.py --config config.yaml

#-----------------------------------------------------------------------
# Job Completion
#-----------------------------------------------------------------------
echo "Training script finished."
echo "Job ended at $(date)"

exit 0