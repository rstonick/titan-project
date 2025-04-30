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

accelerate launch /home/hice1/rstonick3/scratch/titan-project/evaluate_base_model.py \
    --config final_configs/config_run_17_mlp_only_r8.yaml \
    --eval_batch_size 8 \
    --wandb_project "Model Evals f1, perplexity + more" \
    --wandb_entity "ryno-georgia-institute-of-technology" \
    --wandb_run_name "eval_mistral_7b_base" 