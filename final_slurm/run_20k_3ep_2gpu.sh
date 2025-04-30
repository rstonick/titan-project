#!/bin/bash

#SBATCH --job-name=20k_3ep_2gpu          # Job name reflecting parameters (Updated)
#SBATCH --gres=gpu:h200:2              # Request 2 H200 GPUs (Updated)
#SBATCH -N 1                           # Request 1 node
#SBATCH --ntasks-per-node=2            # Number of tasks (match GPU count for DDP) (Updated)
#SBATCH --cpus-per-task=16             # CPUs per task
#SBATCH --mem=250G                     # Request system RAM
#SBATCH --time=3:00:00                 # Time limit: 4 hours (adjust based on testing)
#SBATCH -o /home/hice1/rstonick3/scratch/slurm_logs/slurm_%j.out  # Standard output log file
#SBATCH -e /home/hice1/rstonick3/scratch/slurm_logs/slurm_%j.err  # Standard error log file

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Setting up environment..."
module purge                  # Clear existing modules
module load anaconda3/2023.03 # Load Anaconda
module load cuda/12.1.1       # Load CUDA

# Activate your specific Conda environment
echo "Activating Conda environment: /home/hice1/rstonick3/scratch/titan_env_new"
source activate /home/hice1/rstonick3/scratch/titan_env_new

# Set Hugging Face cache directories to scratch space
echo "Setting Hugging Face cache environment variables..."
export HF_HOME=/home/hice1/rstonick3/scratch/cache/huggingface
export HF_DATASETS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/hub
echo "HF_HOME set to: $HF_HOME"

# Navigate to the directory containing your training script and code/ subdir
echo "Navigating to code directory: /home/hice1/rstonick3/scratch/titan-project/"
cd /home/hice1/rstonick3/scratch/titan-project/
echo "Current directory: $(pwd)"

# --- Run the Training Script using accelerate ---
echo "Starting Python script via accelerate launch: train_mistral_lora_accelerate.py"
# References config file in final_configs/
# Explicitly set accelerate parameters to match SLURM request and config
accelerate launch --num_processes=2 --num_machines=1 --mixed_precision=bf16 train_mistral_lora_accelerate.py --config final_configs/config_20k_3epoch_multiGPU.yaml

# Capture exit status
EXIT_STATUS=$?
echo "Accelerate launch finished with exit status: $EXIT_STATUS"

# --- Cleanup ---
echo "Deactivating environment..."
conda deactivate

echo "Job finished at: $(date)"

exit $EXIT_STATUS
