#!/bin/bash

#SBATCH --job-name=triviaqa_eval_array # Job name
#SBATCH --partition=coe-gpu            # Specify the partition
#SBATCH --array=1-38                 # Array size based on ~38 run directories found
#SBATCH --gres=gpu:h200:1              # Request 1 GPU for inference
#SBATCH -N 1                           # Request 1 node per task
#SBATCH --ntasks-per-node=1            # Number of tasks
#SBATCH --cpus-per-task=8              # CPUs per task (can likely be reduced from training)
#SBATCH --mem=80G                       # Memory per task (can likely be reduced from training)
#SBATCH --time=1:00:00                 # Time limit per task (adjust as needed)
#SBATCH -o /home/hice1/rstonick3/scratch/slurm_logs/eval_slurm_%A_%a.out  # Standard output log file
#SBATCH -e /home/hice1/rstonick3/scratch/slurm_logs/eval_slurm_%A_%a.err  # Standard error log file

# --- Run Directory Mapping ---
# Map SLURM_ARRAY_TASK_ID to run directory names in final_outputs
declare -A run_map
# Add entries based on the listing of /home/hice1/rstonick3/scratch/titan-project/final_outputs
run_map[1]="run_1_q_only"
run_map[2]="run_2_k_only"
run_map[3]="run_3_v_only"
run_map[4]="run_4_o_only"
run_map[5]="run_5_qk"
run_map[6]="run_6_qv"
run_map[7]="run_7_qo"
run_map[8]="run_8_qkv_r2"
run_map[9]="run_9_qkv_r4"
run_map[10]="run_10_qkv_r8"
run_map[11]="run_11_qkv_r16"
run_map[12]="run_12_qkv_r32"
run_map[13]="run_13_qkv_r64"
run_map[14]="run_14_pattern_v_high"
run_map[15]="run_15_pattern_all_layers"
run_map[16]="run_16_pattern_all_layers_r16"
run_map[17]="run_17_mlp_only_r8"
run_map[18]="run_18_mlp_only_r16"
run_map[19]="run_19_mlp_only_r32"
run_map[20]="run_20_ffn_layers_0_9"
run_map[21]="run_21_ffn_layers_10_21"
run_map[22]="run_22_ffn_layers_22_31"
run_map[23]="run_23_mlp_low_r_all"
run_map[24]="run_24_mlp_high_r_all"
run_map[25]="run_25_mlp_med_r_gate_up"
run_map[26]="run_26_mlp_med_r_down_high_dropout"
run_map[27]="run_27_mlp_high_r_all_no_dropout"
run_map[28]="run_28_mlp_med_r_all_high_alpha"
run_map[29]="siavash_15"
run_map[30]="siavash_config_09"
run_map[31]="siavash_config_11"
run_map[32]="siavash_config_12"
run_map[33]="siavash_config_13"
run_map[34]="siavash_config_14"
# Add the numbered directories if they represent valid runs
run_map[35]="5"
run_map[36]="6"
run_map[37]="7"
run_map[38]="8"
run_map[39]="9" # Example if more exist
run_map[40]="10"

# Get the specific run directory for the current array task ID
RUN_DIR_NAME=${run_map[$SLURM_ARRAY_TASK_ID]}

# Check if run directory name is found
if [ -z "$RUN_DIR_NAME" ]; then
  echo "Error: Run directory name not found for task ID $SLURM_ARRAY_TASK_ID."
  exit 1
fi

# Define Paths
PROJECT_DIR="/home/hice1/rstonick3/scratch/titan-project"
OUTPUT_BASE_DIR="${PROJECT_DIR}/final_outputs"
RUN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_DIR_NAME}"
ADAPTER_PATH="${RUN_OUTPUT_DIR}/final_model" # Assuming adapters are saved in 'final_model'
PREDICTION_FILE="${RUN_OUTPUT_DIR}/predictions_${RUN_DIR_NAME}.json"

# !!! IMPORTANT: REPLACE WITH ACTUAL PATH TO GROUND TRUTH FILE !!!
GROUND_TRUTH_FILE="/path/to/original/triviaqa_test.json"

# Check if adapter path exists
if [ ! -d "$ADAPTER_PATH" ]; then
  echo "Error: Adapter path not found for run ${RUN_DIR_NAME}: ${ADAPTER_PATH}"
  exit 1
fi

# Check if ground truth file exists
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
  echo "Error: Ground truth dataset file not found: ${GROUND_TRUTH_FILE}"
  echo "Please update the GROUND_TRUTH_FILE variable in the script."
  exit 1
fi

echo "SLURM Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing Run Directory: $RUN_DIR_NAME"
echo "Adapter Path: $ADAPTER_PATH"
echo "Prediction File: $PREDICTION_FILE"
echo "Ground Truth File: $GROUND_TRUTH_FILE"

# --- Environment Setup ---
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Setting up environment..."
module purge
module load anaconda3/2023.03
module load cuda/12.1.1

echo "Activating Conda environment: /home/hice1/rstonick3/scratch/titan_env_new"
source activate /home/hice1/rstonick3/scratch/titan_env_new

# Set Hugging Face cache directories
echo "Setting Hugging Face cache environment variables..."
export HF_HOME="/home/hice1/rstonick3/scratch/cache/huggingface"
export HF_DATASETS_CACHE="/home/hice1/rstonick3/scratch/hf_datasets_cache"
echo "HF_HOME set to: $HF_HOME"

# Navigate to the project directory
echo "Navigating to code directory: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"
echo "Current directory: $(pwd)"

# --- Step 1: Generate Predictions ---
echo "Starting prediction generation script: generate_predictions.py"
python generate_predictions.py \
    --adapter_path "${ADAPTER_PATH}" \
    --output_file "${PREDICTION_FILE}" \
    --dataset_name "trivia_qa" \
    --dataset_config "rc.nocontext" \
    --dataset_split "test" \
    --batch_size 16 \ # Adjust batch size based on GPU memory
    --wandb_project "triviaqa-evaluation" \
    --wandb_run_name "${RUN_DIR_NAME}-predict"

# Capture exit status for prediction generation
PREDICT_EXIT_STATUS=$?
if [ $PREDICT_EXIT_STATUS -ne 0 ]; then
    echo "Error: Prediction generation failed for ${RUN_DIR_NAME} with exit status ${PREDICT_EXIT_STATUS}."
    exit $PREDICT_EXIT_STATUS
fi
echo "Prediction generation finished successfully."

# --- Step 2: Run Evaluation ---
echo "Starting evaluation script: triviaqa/evaluation/triviaqa_evaluation.py"
python triviaqa/evaluation/triviaqa_evaluation.py \
    --dataset_file "${GROUND_TRUTH_FILE}" \
    --prediction_file "${PREDICTION_FILE}" \
    --log_to_wandb \
    --wandb_project "triviaqa-evaluation" \
    --wandb_run_name "${RUN_DIR_NAME}-eval"

# Capture exit status for evaluation
EVAL_EXIT_STATUS=$?
if [ $EVAL_EXIT_STATUS -ne 0 ]; then
    echo "Error: Evaluation failed for ${RUN_DIR_NAME} with exit status ${EVAL_EXIT_STATUS}."
    exit $EVAL_EXIT_STATUS
fi
echo "Evaluation finished successfully."

# --- Cleanup ---
echo "Deactivating environment..."
conda deactivate

echo "Job task $SLURM_ARRAY_TASK_ID ($RUN_DIR_NAME) finished at: $(date)"

exit 0
