#!/bin/bash

# Set Hugging Face cache directories to scratch space
echo "Setting Hugging Face cache environment variables..."
export HF_HOME="/home/hice1/rstonick3/scratch/cache/huggingface"
export HF_DATASETS_CACHE="/home/hice1/rstonick3/scratch/hf_datasets_cache"
# Silence tokenizer parallelism warning
export TOKENIZERS_PARALLELISM=false
echo "HF_HOME set to: $HF_HOME"
echo "HF_DATASETS_CACHE set to: $HF_DATASETS_CACHE"
echo "TOKENIZERS_PARALLELISM set to: $TOKENIZERS_PARALLELISM"

# --- Add project directory to PYTHONPATH ---
PROJECT_DIR="/home/hice1/rstonick3/scratch/titan-project"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" # Prepend project dir to existing PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# Activate Conda environment (Uncomment if needed for your setup)
# echo "Activating Conda environment..."
# source activate /home/hice1/rstonick3/scratch/titan_env_new

# --- Remove the cd command ---
# echo "Changing directory to: $PROJECT_DIR"
# cd "$PROJECT_DIR" || exit 1 # No longer needed

# The actual command to run the evaluation
# Use absolute paths
echo "Launching evaluation script..."
accelerate launch "$PROJECT_DIR/evaluate_models.py" \
    --config "$PROJECT_DIR/final_configs/eval_config.yaml"

echo "Evaluation script finished."
