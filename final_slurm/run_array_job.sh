#!/bin/bash

#SBATCH --job-name=mistral_lora_array # Job name
#SBATCH --array=23-25                # Submit tasks for runs 23 through 28
#SBATCH --gres=gpu:h200:1              # Request 1 H200 GPU per task
#SBATCH -N 1                           # Request 1 node per task
#SBATCH --ntasks-per-node=1            # Number of tasks (Changed from 2)
#SBATCH --cpus-per-task=12             # CPUs per task (Kept 16, can potentially reduce later)
#SBATCH --mem=100G                      # Request system RAM per task (Reduced from 350G)
#SBATCH --time=6:00:00                 # Time limit per task (Increased from 3:00:00)
#SBATCH -o /home/hice1/rstonick3/scratch/slurm_logs/slurm_%A_%a.out  # Standard output log file (%A=jobID, %a=taskID)
#SBATCH -e /home/hice1/rstonick3/scratch/slurm_logs/slurm_%A_%a.err  # Standard error log file (%A=jobID, %a=taskID)

# --- Configuration Mapping ---
# !!! IMPORTANT: EDIT THIS SECTION TO MATCH YOUR EXACT CONFIG FILENAMES !!!
declare -A config_map
# Updated based on directory listing
#config_map[1]="config_run_1_q_only.yaml"
#config_map[2]="config_run_2_k_only.yaml"
#config_map[3]="config_run_3_v_only.yaml"
#config_map[4]="config_run_4_o_only.yaml"
#config_map[5]="config_run_5_qk.yaml"
#config_map[6]="config_run_6_qv.yaml"
#config_map[7]="config_run_7_qo.yaml" # Corrected
#config_map[8]="config_run_8_qkv_r2.yaml" # Corrected
# config_map[9]="config_run_9_qkv_r4.yaml" # Corrected
# config_map[10]="config_run_10_qkv_r8.yaml" # Corrected
# config_map[11]="config_run_11_qkv_r16.yaml" # Corrected
# config_map[12]="config_run_12_qkv_r32.yaml" # Corrected
# config_map[13]="config_run_13_qkv_r64.yaml" # Corrected
# config_map[14]="config_run_14_pattern_v_high.yaml" # Corrected
# config_map[15]="config_run_15_pattern_all_layers.yaml" # Corrected
# config_map[16]="config_run_16_pattern_all_layers_r16.yaml" # Corrected
# config_map[17]="config_run_17_mlp_only_r8.yaml" # Corrected
# config_map[18]="config_run_18_mlp_only_r16.yaml" # Corrected
# config_map[19]="config_run_19_mlp_only_r32.yaml" # Corrected
# config_map[20]="config_run_20_ffn_layers_0_9.yaml"
# config_map[21]="config_run_21_ffn_layers_10_21.yaml"
# config_map[22]="config_run_22_ffn_layers_22_31.yaml"
# Add entries for runs 23-28
config_map[24]="config_siavash_17.yaml"
config_map[25]="config_siavash_18.yaml"

# Get the specific config file for the current array task ID
CONFIG_FILE_NAME=${config_map[$SLURM_ARRAY_TASK_ID]}
# Assuming config files are in ../final_configs relative to this script's location
# Adjust path if necessary - Changed to absolute path for robustness
CONFIG_PATH="/home/hice1/rstonick3/scratch/titan-project/final_configs/${CONFIG_FILE_NAME}" # Using absolute path

# Check if config file exists
if [ -z "$CONFIG_FILE_NAME" ] || [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found for task ID $SLURM_ARRAY_TASK_ID."
  echo "Looked for: $CONFIG_PATH"
  exit 1
fi

echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using Config File: $CONFIG_PATH"

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Setting up environment..."
module purge                  # Clear existing modules
module load anaconda3/2023.03 # Reverted to previous version
module load cuda/12.1.1

# Activate your specific Conda environment
echo "Activating Conda environment: /home/hice1/rstonick3/scratch/titan_env_new"
source activate /home/hice1/rstonick3/scratch/titan_env_new

# Set Hugging Face cache directories to scratch space
echo "Setting Hugging Face cache environment variables..."
export HF_HOME="/home/hice1/rstonick3/scratch/cache/huggingface"
export HF_DATASETS_CACHE="/home/hice1/rstonick3/scratch/hf_datasets_cache"
echo "HF_HOME set to: $HF_HOME"

# --- Set NCCL Environment Variable for Timeout Mitigation ---
#echo "Setting NCCL_BLOCKING_WAIT=1 to potentially mitigate timeouts..."
#export NCCL_BLOCKING_WAIT=1
# echo "Setting NCCL_P2P_DISABLE=1..." # Commented out to allow P2P
# export NCCL_P2P_DISABLE=1
# echo "Setting NCCL_ASYNC_ERROR_HANDLING=1..." # Commented out old variable
# export NCCL_ASYNC_ERROR_HANDLING=1
echo "Setting TORCH_NCCL_ASYNC_ERROR_HANDLING=1..." # Added recommended variable
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "Setting NCCL_DEBUG=WARN..." # Changed from INFO
export NCCL_DEBUG=WARN
# export TORCH_DISTRIBUTED_DEBUG=DETAIL # Commented out

# --- End NCCL Environment Variable ---

# Navigate to the directory containing your training script and code/ subdir
echo "Navigating to code directory: /home/hice1/rstonick3/scratch/titan-project/"
cd /home/hice1/rstonick3/scratch/titan-project/
echo "Current directory: $(pwd)"

# --- Distributed Training Setup ---
# Get the first node allocated to this job task
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# Get a random free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
# --- End Distributed Training Setup ---


# --- Run the Training Script using accelerate ---
echo "Starting Python script via accelerate launch: train_mistral_lora_accelerate.py with config $CONFIG_PATH"
# Use the dynamically selected config path
# Explicitly pass the MASTER_PORT to accelerate launch (though less critical for 1 GPU)
# Changed num_processes to 1
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --main_process_port $MASTER_PORT train_mistral_lora_accelerate.py --config "$CONFIG_PATH" # Changed num_processes to 1

# Capture exit status
EXIT_STATUS=$?
echo "Accelerate launch finished for task $SLURM_ARRAY_TASK_ID with exit status: $EXIT_STATUS"

# --- Cleanup ---
echo "Deactivating environment..."
conda deactivate

echo "Job task $SLURM_ARRAY_TASK_ID finished at: $(date)"

exit $EXIT_STATUS
