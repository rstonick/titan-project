#!/bin/bash

#SBATCH --job-name=config_array       # Job name
#SBATCH --gres=gpu:h200:1           # Request 1 H200 GPU per job task
#SBATCH -N 1                        # Request 1 node per job task
#SBATCH --ntasks-per-node=1         # Number of tasks (match GPU count)
#SBATCH --cpus-per-task=12          # CPUs per task
#SBATCH --mem=100G                  # Request system RAM
#SBATCH --time=4:00:00              # Time limit: 3 hours (adjust as needed)
#SBATCH -o /home/hice1/rstonick3/scratch/slurm_logs/slurm_array_%A_%a.out  # Standard output log (%A=jobID, %a=taskID)
#SBATCH -e /home/hice1/rstonick3/scratch/slurm_logs/slurm_array_%A_%a.err  # Standard error log (%A=jobID, %a=taskID)

# --- Configuration ---
CONFIG_DIR="/home/hice1/rstonick3/scratch/titan-project/final_configs"
SCRIPT_PATH="/home/hice1/rstonick3/scratch/titan-project/train_mistral_lora_accelerate_refactored.py"
PROJECT_DIR="/home/hice1/rstonick3/scratch/titan-project"
CONDA_ENV_PATH="/home/hice1/rstonick3/scratch/titan_env_new"
TEMP_CONFIG_DIR="/home/hice1/rstonick3/scratch/temp_configs" # Directory for temporary configs

# --- Find config files and set array size ---
# Use find for safer handling of filenames, sort for consistent ordering
mapfile -t CONFIG_FILES < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -print | sort)
NUM_CONFIGS=${#CONFIG_FILES[@]}

if [ "$NUM_CONFIGS" -eq 0 ]; then
  echo "Error: No YAML config files found in $CONFIG_DIR"
  exit 1
fi

# Dynamically set the array size in the submission command, e.g., sbatch --array=1-$NUM_CONFIGS run_configs_array.slurm
# Or uncomment and hardcode if submitting directly:
# #SBATCH --array=1-1 # Example: Set manually if needed, replace 1 with $NUM_CONFIGS

echo "Found $NUM_CONFIGS config files."
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"

# --- Get the specific config file for this task ---
# SLURM_ARRAY_TASK_ID is 1-based, array indices are 0-based
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$TASK_INDEX" -lt 0 ] || [ "$TASK_INDEX" -ge "$NUM_CONFIGS" ]; then
  echo "Error: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (must be between 1 and $NUM_CONFIGS)"
  exit 1
fi
ORIGINAL_CONFIG_PATH="${CONFIG_FILES[$TASK_INDEX]}"
CONFIG_BASENAME=$(basename "$ORIGINAL_CONFIG_PATH" .yaml)

echo "Processing original config: $ORIGINAL_CONFIG_PATH"

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Setting up environment..."
module purge                  # Clear existing modules
module load anaconda3/2022.05.0.1
module load cuda/12.1.1

# Activate Conda environment
echo "Activating Conda environment: $CONDA_ENV_PATH"
source activate "$CONDA_ENV_PATH"

# Set Hugging Face cache directories
echo "Setting Hugging Face cache environment variables..."
export HF_HOME=/home/hice1/rstonick3/scratch/cache/huggingface
export HF_DATASETS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/home/hice1/rstonick3/scratch/cache/huggingface/hub
echo "HF_HOME set to: $HF_HOME"

# Disable tokenizer parallelism
echo "Setting TOKENIZERS_PARALLELISM=false"
export TOKENIZERS_PARALLELISM=false

# --- Create Temporary Overridden Config ---
# Define overrides here (adjust values as needed)
# Using a heredoc for the Python script for clarity
OVERRIDE_PYTHON_SCRIPT=$(cat <<EOF
import yaml
import sys
import os

original_config_path = "$ORIGINAL_CONFIG_PATH"
temp_config_path = "$TEMP_CONFIG_DIR/${CONFIG_BASENAME}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.yaml"

# Define the overrides
overrides = {

    'wandb': {
        'project': "Final late night run EM F1 and perplexity" # Adjusted project name
    },
    'dataset': {
        'local_path': '/home/hice1/rstonick3/scratch/triviaqa_dataset',
        'train_files': [
            'qa/wikipedia-train.json',
            'qa/web-train.json'  # Add web training data
        ],
        'validation_files': [
            'qa/wikipedia-dev.json',
            'qa/web-dev.json'    # Add web validation data
        ],
        'subset_size': 20000, # Keep consistent subset size or adjust as needed
        'prompt_format': "Question: {question}\\\\nAnswer: {answer_value}"
    },
    'training': {
        'batch_size': 128,    # Override batch size
        'max_length': 512,     # Override max length
        'num_epochs': 3,        # Override number of epochs
        'per_device_eval_batch_size': 16,
        # learning_rate, num_epochs etc. will be taken from original unless overridden here
    },
    'trainer': {
        'evaluation_strategy': "epoch" # Override evaluation strategy
        # logging_steps, save_strategy etc. will be taken from original unless overridden here
    }
    # Add other top-level keys like 'model', 'quantization', 'lora' if they need overrides
    # Note: This simple merge won't deeply merge nested dicts other than the ones specified.
    # For example, it replaces the *entire* 'training' dict if it exists, then adds batch_size/max_length.
    # A more robust deep merge might be needed for complex overrides within existing sections.
}

try:
    # Load original config
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides (simple top-level merge, then specific nested ones)
    # This replaces the entire 'dataset' section
    config['dataset'] = overrides['dataset']

    # Update 'training' section - ensure it exists
    if 'training' not in config:
        config['training'] = {}
    config['training']['batch_size'] = overrides['training']['batch_size']
    config['training']['max_length'] = overrides['training']['max_length']

    # Update 'trainer' section - ensure it exists
    if 'trainer' not in config:
        config['trainer'] = {}
    config['trainer']['evaluation_strategy'] = overrides['trainer']['evaluation_strategy']

    # Ensure temp directory exists
    os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)

    # Save the modified config
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Print the path of the temporary config file for the bash script
    print(temp_config_path)

except Exception as e:
    print(f"Error processing config {original_config_path}: {e}", file=sys.stderr)
    sys.exit(1)

EOF
)

echo "Creating temporary config file..."
# Create the directory for temporary files if it doesn't exist
mkdir -p "$TEMP_CONFIG_DIR"

# Execute the Python script to create the temp config and capture its path
TEMP_CONFIG_PATH=$(python -c "$OVERRIDE_PYTHON_SCRIPT")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ -z "$TEMP_CONFIG_PATH" ] || [ ! -f "$TEMP_CONFIG_PATH" ]; then
  echo "Error: Failed to create temporary config file for $ORIGINAL_CONFIG_PATH."
  # Attempt cleanup just in case
  rm -f "$TEMP_CONFIG_PATH"
  exit 1
fi

echo "Temporary config created at: $TEMP_CONFIG_PATH"

# --- Setup Cleanup Trap ---
# Ensure temporary file is deleted even if the script fails
trap 'echo "Cleaning up temporary config: $TEMP_CONFIG_PATH"; rm -f "$TEMP_CONFIG_PATH"' EXIT

# --- Navigate to Project Directory ---
echo "Navigating to code directory: $PROJECT_DIR"
cd "$PROJECT_DIR"
echo "Current directory: $(pwd)"

# --- Run the Training Script using accelerate ---
echo "Starting Python script via accelerate launch using temp config: $TEMP_CONFIG_PATH"

accelerate launch "$SCRIPT_PATH" --config "$TEMP_CONFIG_PATH"

# Capture exit status
EXIT_STATUS=$?
echo "Accelerate launch finished with exit status: $EXIT_STATUS"

# --- Cleanup (handled by trap) ---
echo "Deactivating environment..."
conda deactivate

echo "Job finished at: $(date)"

exit $EXIT_STATUS