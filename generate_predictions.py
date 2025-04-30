# /home/hice1/rstonick3/scratch/titan-project/generate_predictions.py
import argparse
import os
import sys
import json

import torch
import wandb

# Add project root to sys.path to allow importing triviaqa.utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Now import dataset_utils
from triviaqa.utils import dataset_utils
from triviaqa.utils import utils # Also ensure utils is available if needed by dataset_utils

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate predictions for TriviaQA using a fine-tuned Mistral model.")
    parser.add_argument('--base_model_name', type=str, default="mistralai/Mistral-7B-v0.1", help='Base model ID')
    parser.add_argument('--adapter_path', type=str, required=False, default=None, help='Optional path to the trained LoRA adapters (final_model directory). If not provided, only the base model is used.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the predictions JSON file')
    parser.add_argument('--dataset_name', type=str, default="trivia_qa", help='Dataset name')
    parser.add_argument('--dataset_config', type=str, default="rc.nocontext", help='Dataset configuration')
    parser.add_argument('--dataset_split', type=str, default="test", help='Dataset split to use (e.g., test, validation)')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate for the answer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference') # Added batch size argument
    # --- WandB Arguments ---
    parser.add_argument('--wandb_project', type=str, default="triviaqa-evaluation", help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (defaults to output filename base)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- WandB Initialization ---
    wandb_run_name = args.wandb_run_name or os.path.splitext(os.path.basename(args.output_file))[0]
    try:
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args) # Log command line arguments
        )
        print(f"WandB run initialized: {wandb.run.name} (Project: {args.wandb_project})")
    except Exception as e:
        print(f"Warning: Could not initialize WandB. {e}")
        wandb.init(mode="disabled") # Ensure wandb calls don't crash

    # --- Quantization Config ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load Base Model ---
    print(f"Loading base model: {args.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=quantization_config,
        device_map="auto", # Automatically distribute across available GPUs
        trust_remote_code=True, # Added based on potential model requirements
        attn_implementation="flash_attention_2" # Enable Flash Attention 2
    )
    print("Base model loaded.")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer: {args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token for batching
    print("Tokenizer loaded.")

    # --- Load LoRA Adapters (Optional) ---
    if args.adapter_path:
        print(f"Loading LoRA adapters from: {args.adapter_path}")
        if not os.path.exists(args.adapter_path):
            print(f"Error: Adapter path not found: {args.adapter_path}")
            wandb.finish(exit_code=1)
            return
        try:
            model = PeftModel.from_pretrained(model, args.adapter_path)
            model = model.merge_and_unload() # Merge adapters for faster inference
            print("LoRA adapters loaded and merged.")
        except Exception as e:
            print(f"Error loading LoRA adapters: {e}")
            wandb.finish(exit_code=1)
            return
    else:
        print("No adapter_path provided. Using base model only.")

    # --- Load Dataset using dataset_utils.read_triviaqa_data ---
    ground_truth_path = "/home/hice1/rstonick3/scratch/triviaqa_dataset/qa/verified-web-dev.json"
    print(f"Loading and filtering questions using dataset_utils.read_triviaqa_data from: {ground_truth_path}")
    try:
        # Use the official utility function to read and filter the data
        triviaqa_filtered_data = dataset_utils.read_triviaqa_data(ground_truth_path)

        # Extract questions and their IDs from the filtered data
        questions_data = triviaqa_filtered_data.get('Data', []) # read_triviaqa_data should have already filtered this list
        print(f"[DEBUG] dataset_utils.read_triviaqa_data returned {len(questions_data)} items.")
        if not questions_data:
            raise ValueError("dataset_utils.read_triviaqa_data returned no data.")
        # No need to check for 325 here, we accept the result of read_triviaqa_data

        question_ids = []
        questions = []
        # Need to handle the complex key format from web-dev.json for IDs
        for item in questions_data: # Iterate through the filtered list (e.g., 322 items)
            q_id = item.get('QuestionId')
            # If the file uses complex keys like qid--evidence.txt, extract base qid
            base_q_id = q_id.split('--')[0] if q_id and '--' in q_id else q_id
            q_text = item.get('Question')
            if base_q_id and q_text:
                question_ids.append(base_q_id) # Store the base ID
                questions.append(q_text)
            else:
                print(f"Warning: Skipping item from filtered data due to missing QuestionId or Question: {item}")

        if not question_ids:
             raise ValueError("No valid questions with IDs found in the filtered data.")

        print(f"Successfully loaded and processed {len(questions)} questions after filtering.")
        wandb.config.update({"dataset_size": len(questions)}) # Log dataset size

    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        wandb.finish(exit_code=1)
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {ground_truth_path}: {e}")
        wandb.finish(exit_code=1)
        return
    except ImportError as e:
         print(f"Error importing triviaqa utils: {e}. Check sys.path and file locations.")
         wandb.finish(exit_code=1)
         return
    except Exception as e:
        print(f"Error processing ground truth file {ground_truth_path} using dataset_utils: {e}")
        wandb.finish(exit_code=1)
        return

    # --- Prepare for Inference ---
    model.eval() # Set model to evaluation mode
    predictions = {}
    # question_ids and questions lists are now populated from the JSON file

    # --- Generate Predictions (with batching) ---
    print(f"Generating predictions with batch size {args.batch_size}...")
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_qids = question_ids[i:i+args.batch_size]
        batch_questions = questions[i:i+args.batch_size]

        # Simple prompt format: "Question: [question text] Answer:"
        # Adjust this if your fine-tuning used a different format
        prompts = [f"Question: {q} Answer:" for q in batch_questions]

        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device) # Ensure tensors are on the same device as the model

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Use greedy decoding for deterministic output
                pad_token_id=tokenizer.pad_token_id # Ensure pad token ID is set
            )

        # Decode and store predictions
        # Decode only the generated part (excluding the prompt)
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_answers = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for qid, answer in zip(batch_qids, decoded_answers):
             # Basic post-processing: remove leading/trailing whitespace
            predictions[qid] = answer.strip()

    # --- Save Predictions ---
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Saving {len(predictions)} predictions to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    # --- Log Artifact to WandB ---
    try:
        artifact = wandb.Artifact(name=f"predictions-{wandb.run.id}", type="predictions")
        artifact.add_file(args.output_file)
        wandb.log_artifact(artifact)
        print(f"Logged predictions artifact to WandB: {artifact.name}")
    except Exception as e:
        print(f"Warning: Could not log predictions artifact to WandB. {e}")


    print("Prediction generation complete.")
    wandb.finish() # Finish the wandb run

if __name__ == "__main__":
    main()
