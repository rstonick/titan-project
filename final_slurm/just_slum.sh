#!/bin/bash

#SBATCH --job-name=17          # Job name reflecting parameters (Updated for 1 GPU)
#SBATCH --gres=gpu:h200:1              # Request 1 H200 GPU (Updated)
#SBATCH -N 1                           # Request 1 node
#SBATCH --ntasks-per-node=1            # Number of tasks (match GPU count) (Updated)
#SBATCH --cpus-per-task=34             # CPUs per task (Kept same, adjust if needed)
#SBATCH --mem=260G                     # Request system RAM (Kept same, adjust if needed)
#SBATCH --time=3:00:00                 # Time limit: 3 hours (adjust based on testing)
#SBATCH -o /home/hice1/rstonick3/scratch/slurm_logs/slurm_%j.out  # Standard output log file
#SBATCH -e /home/hice1/rstonick3/scratch/slurm_logs/slurm_%j.err  # Standard error log file

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Setting up environment..."
module purge                  # Clear existing modules
module load anaconda3/2022.05.0.1  # Reverted to previous version
module load cuda/12.1.1

# Activate your specific Conda environment
echo "Activating Conda environment: /home/hice1/rstonick3/scratch/titan_env_new"
source activate /home/hice1/rstonick3/scratch/titan_env_new

# Set Hugging Face cache directories to scratch space
echo "Setting Hugging Face cache environment variables..."
export HF_HOME=/home/hice1/rstonick3/scratch/cache/huggingface
export HF_DATASETS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/hub
echo "HF_HOME set to: $HF_HOME"

# Disable tokenizer parallelism to avoid warnings/potential deadlocks in forks
echo "Setting TOKENIZERS_PARALLELISM=false"
export TOKENIZERS_PARALLELISM=false

# Navigate to the directory containing your training script and code/ subdir
echo "Navigating to code directory: /home/hice1/rstonick3/scratch/titan-project/"
cd /home/hice1/rstonick3/scratch/titan-project/
echo "Current directory: $(pwd)"

# --- Run the Training Script using accelerate ---
echo "Starting Python script via accelerate launch: train_mistral_lora_accelerate.py"
# References config file in final_configs/
# Explicitly set accelerate parameters to match SLURM request and config
accelerate launch /home/hice1/rstonick3/scratch/titan-project/train_mistral_lora_accelerate_refactored.py --config /home/hice1/rstonick3/scratch/titan-project/final_slurm/this.yaml
# Capture exit status
EXIT_STATUS=$?
echo "Accelerate launch finished with exit status: $EXIT_STATUS"

# --- Cleanup ---
echo "Deactivating environment..."
conda deactivate

echo "Job finished at: $(date)"

exit $EXIT_STATUS