#!/bin/bash

# Script to run prediction generation and evaluation locally, one run after another.

# --- Configuration ---
PROJECT_DIR="/home/hice1/rstonick3/scratch/titan-project"
OUTPUT_BASE_DIR="${PROJECT_DIR}/final_outputs"

# !!! IMPORTANT: REPLACE WITH ACTUAL PATH TO GROUND TRUTH FILE !!!
GROUND_TRUTH_FILE="/path/to/original/triviaqa_test.json"

# Optional: Activate Conda environment if needed
# echo "Activating Conda environment..."
# source activate /home/hice1/rstonick3/scratch/titan_env_new

# Optional: Set Hugging Face cache directories if needed
# export HF_HOME="/home/hice1/rstonick3/scratch/cache/huggingface"
# export HF_DATASETS_CACHE="/home/hice1/rstonick3/scratch/hf_datasets_cache"

# Check if ground truth file exists
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
  echo "Error: Ground truth dataset file not found: ${GROUND_TRUTH_FILE}"
  echo "Please update the GROUND_TRUTH_FILE variable in the script."
  exit 1
fi

# Navigate to the project directory
cd "${PROJECT_DIR}" || exit 1
echo "Changed directory to: $(pwd)"

# --- Loop Through Run Directories ---
# Find all directories within the output base directory
# Use find to handle potential spaces or special characters in names, though unlikely here.
find "${OUTPUT_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r RUN_OUTPUT_DIR; do
    RUN_DIR_NAME=$(basename "${RUN_OUTPUT_DIR}")
    echo "-----------------------------------------------------"
    echo "Processing Run: ${RUN_DIR_NAME}"
    echo "-----------------------------------------------------"

    ADAPTER_PATH="${RUN_OUTPUT_DIR}/final_model"
    PREDICTION_FILE="${RUN_OUTPUT_DIR}/predictions_${RUN_DIR_NAME}.json"

    # Check if adapter path exists
    if [ ! -d "$ADAPTER_PATH" ]; then
      echo "Warning: Adapter path not found for run ${RUN_DIR_NAME}, skipping: ${ADAPTER_PATH}"
      continue # Skip to the next directory
    fi

    # --- Step 1: Generate Predictions ---
    echo "Starting prediction generation for ${RUN_DIR_NAME}..."
    python generate_predictions.py \
        --adapter_path "${ADAPTER_PATH}" \
        --output_file "${PREDICTION_FILE}" \
        --dataset_name "trivia_qa" \
        --dataset_config "rc.nocontext" \
        --dataset_split "test" \
        --batch_size 16 \ # Adjust batch size based on local GPU memory
        --wandb_project "triviaqa-evaluation-local" \
        --wandb_run_name "${RUN_DIR_NAME}-predict"

    PREDICT_EXIT_STATUS=$?
    if [ $PREDICT_EXIT_STATUS -ne 0 ]; then
        echo "Error: Prediction generation failed for ${RUN_DIR_NAME} with exit status ${PREDICT_EXIT_STATUS}."
        # Decide whether to stop or continue with the next run
        # continue # Uncomment to continue with the next run despite error
        # exit $PREDICT_EXIT_STATUS # Uncomment to stop the entire script on error
    else
        echo "Prediction generation finished successfully for ${RUN_DIR_NAME}."

        # --- Step 2: Run Evaluation (only if prediction succeeded) ---
        echo "Starting evaluation for ${RUN_DIR_NAME}..."
        python triviaqa/evaluation/triviaqa_evaluation.py \
            --dataset_file "${GROUND_TRUTH_FILE}" \
            --prediction_file "${PREDICTION_FILE}" \
            --log_to_wandb \
            --wandb_project "triviaqa-evaluation-local" \
            --wandb_run_name "${RUN_DIR_NAME}-eval"

        EVAL_EXIT_STATUS=$?
        if [ $EVAL_EXIT_STATUS -ne 0 ]; then
            echo "Error: Evaluation failed for ${RUN_DIR_NAME} with exit status ${EVAL_EXIT_STATUS}."
            # Decide whether to stop or continue
            # continue
            # exit $EVAL_EXIT_STATUS
        else
            echo "Evaluation finished successfully for ${RUN_DIR_NAME}."
        fi
    fi

done

echo "-----------------------------------------------------"
echo "All processing complete."

# Optional: Deactivate Conda environment if activated
# echo "Deactivating Conda environment..."
# conda deactivate

exit 0
