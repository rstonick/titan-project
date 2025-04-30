import os
import torch
import math
import numpy as np
from tqdm.auto import tqdm
import logging
import argparse
import wandb

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
# Note: PeftModel is no longer needed for the base model evaluation
from accelerate import Accelerator
from accelerate.logging import get_logger

# Import necessary functions from your project structure
from code.config_loader import load_config
from code.data_preparation import prepare_tokenizer, prepare_dataset
from code.metrics import calculate_qa_metrics, normalize_answer # Make sure normalize_answer is imported

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a base Hugging Face model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file specifying the base model and dataset.')
    # --models_dir is removed as we are evaluating the base model directly
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Override evaluation batch size per device. Defaults to value in config.')
    # Add wandb arguments
    parser.add_argument('--wandb_project', type=str, default="base_model_evaluation", help='Wandb project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team).')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name. Defaults to a generated name (e.g., eval_base_model_name).')
    return parser.parse_args()

def load_model_and_tokenizer(config):
    """Load the base model and tokenizer specified in the config."""
    base_model_name = config['model']['name']
    logger.info(f"Loading base model: {base_model_name}")

    # --- Load Tokenizer ---
    tokenizer = prepare_tokenizer(config) # Uses config['model']['name']
    logger.info(f"Loaded tokenizer for {base_model_name}")

    # --- Load Base Model (with quantization if specified) ---
    quantization_config = None
    if config['model'].get('quantization'):
        logger.info("Applying quantization configuration...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config['model']['quantization'].get('load_in_4bit', True),
            bnb_4bit_quant_type=config['model']['quantization'].get('bnb_4bit_quant_type', "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, config['model']['quantization'].get('bnb_4bit_compute_dtype', "float16")),
            bnb_4bit_use_double_quant=config['model']['quantization'].get('bnb_4bit_use_double_quant', False),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', "float16")),
        device_map="auto", # Let Accelerate handle device placement
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    logger.info(f"Loaded base model {base_model_name}.")

    # --- LoRA Adapter Loading is REMOVED ---

    return model, tokenizer

def prepare_eval_dataloader(tokenizer, config, args):
    """Prepare the evaluation dataloader using the validation split."""
    logger.info("Preparing evaluation dataset...")
    tokenized_dataset = prepare_dataset(tokenizer, config)

    if 'validation' not in tokenized_dataset:
        logger.error("Validation split not found in the prepared dataset. Cannot perform evaluation.")
        raise ValueError("Validation split is required for evaluation.")

    eval_dataset = tokenized_dataset['validation']
    logger.info(f"Using {len(eval_dataset)} examples for evaluation.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Determine batch size
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else config['trainer'].get('per_device_eval_batch_size', config['training']['batch_size'])
    num_workers = config['trainer'].get('dataloader_num_workers', 0)
    logger.info(f"Using evaluation batch size: {eval_batch_size} per device, Num workers: {num_workers}")

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    return eval_dataloader, eval_dataset

def run_evaluation(accelerator, model, tokenizer, eval_dataloader, eval_dataset):
    """Runs the evaluation loop using model.generate() and calculates metrics."""
    model.eval()
    all_preds_decoded = []
    all_labels_decoded = []

    # Define generation config (can be customized)
    # Increased max_new_tokens based on potential answer length
    generation_config = {
        "max_new_tokens": 100,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": True # Stop generating once EOS is reached
    }
    logger.info(f"Generation config: {generation_config}")


    progress_bar = tqdm(total=len(eval_dataloader), desc="Evaluating Base Model (Generation)", disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(eval_dataloader):
        # batch contains 'input_ids', 'attention_mask', 'labels'
        # 'labels' has -100 for prompt tokens, actual token IDs for answer tokens
        # 'input_ids' has the full "Question: ... Answer: ..." sequence

        # Find where the prompt ends and the answer begins in the labels
        # We need the input_ids corresponding to the prompt only
        prompt_input_ids_list = []
        prompt_attention_mask_list = []
        answer_labels_list = []

        for i in range(batch['input_ids'].shape[0]):
            # Find the first non -100 index in labels, this marks the start of the answer
            try:
                # Add 1 because the first answer token is the prediction *after* the last prompt token
                answer_start_index = (batch['labels'][i] != -100).nonzero(as_tuple=True)[0][0].item()
            except IndexError:
                # Handle cases where there might be no answer tokens (shouldn't happen with TriviaQA format)
                logger.warning(f"Skipping example {i} in batch {step} due to no answer tokens found in labels.")
                continue

            # Extract prompt tokens (up to the start of the answer)
            prompt_input_ids = batch['input_ids'][i, :answer_start_index]
            prompt_attention_mask = batch['attention_mask'][i, :answer_start_index]

            # Extract only the answer part of the labels (ignore -100s)
            answer_labels = batch['labels'][i][answer_start_index:]
            # Filter out any remaining -100s (e.g., padding after the answer)
            answer_labels = answer_labels[answer_labels != -100]

            prompt_input_ids_list.append(prompt_input_ids)
            prompt_attention_mask_list.append(prompt_attention_mask)
            answer_labels_list.append(answer_labels)

        if not prompt_input_ids_list: # Skip batch if all examples were problematic
             progress_bar.update(1)
             continue

        # Pad the prompt inputs dynamically for this batch
        prompt_inputs = tokenizer.pad(
            {"input_ids": [ids.tolist() for ids in prompt_input_ids_list]},
            padding=True,
            return_tensors="pt",
        ).to(accelerator.device) # Move padded inputs to the correct device


        with torch.no_grad():
            # Use accelerator.unwrap_model for generation
            unwrapped_model = accelerator.unwrap_model(model)
            generated_outputs = unwrapped_model.generate(
                input_ids=prompt_inputs['input_ids'],
                attention_mask=prompt_inputs['attention_mask'],
                **generation_config
            )

        # Gather generated sequences across devices
        gathered_preds_ids = accelerator.gather_for_metrics(generated_outputs)
        # Gather the ground truth answer labels (need to pad them first for gathering)
        # Max length needed for padding answer labels
        max_label_len = max(len(l) for l in answer_labels_list)
        # Ensure padding tensor is created on the same device as the labels
        padded_labels_list = [torch.cat([l, torch.full((max_label_len - len(l),), tokenizer.pad_token_id, dtype=l.dtype, device=l.device)]) for l in answer_labels_list]
        padded_labels_tensor = torch.stack(padded_labels_list).to(accelerator.device) # Ensure final tensor is on accelerator device just in case
        gathered_labels_ids = accelerator.gather_for_metrics(padded_labels_tensor)


        if accelerator.is_main_process:
            # Decode predictions: Skip prompt tokens and special tokens
            # Prompt length might vary due to left padding, find the actual start of generation
            # For left-padded inputs, the generated part starts right after the padding ends.
            # We can slice generated_outputs based on the length of the prompt_inputs
            prompt_len = prompt_inputs['input_ids'].shape[1]
            decoded_preds_batch = tokenizer.batch_decode(
                gathered_preds_ids[:, prompt_len:], # Slice to get only generated tokens
                skip_special_tokens=True
            )

            # Decode labels: Skip padding tokens
            # Replace pad_token_id with something temporary if it's part of the vocab, then decode
            # gathered_labels_ids[gathered_labels_ids == tokenizer.pad_token_id] = -1 # Use a temporary ID not in vocab if needed
            decoded_labels_batch = tokenizer.batch_decode(
                gathered_labels_ids,
                skip_special_tokens=True
            )

            # Normalize and store for metric calculation
            all_preds_decoded.extend([normalize_answer(p) for p in decoded_preds_batch])
            all_labels_decoded.extend([normalize_answer(l) for l in decoded_labels_batch])

            # --- Debugging Print (Optional) ---
            if step < 2: # Print first 2 batches
                print(f"\n--- Batch {step} Decoded ---")
                for i in range(min(len(decoded_preds_batch), 3)):
                    print(f"Pred {i}: '{decoded_preds_batch[i]}'")
                    print(f"Label {i}: '{decoded_labels_batch[i]}'")
                    print(f"Norm Pred {i}: '{all_preds_decoded[step*len(decoded_preds_batch)+i]}'")
                    print(f"Norm Label {i}: '{all_labels_decoded[step*len(decoded_labels_batch)+i]}'")
                print("--- End Batch ---")
            # --- End Debugging ---


        progress_bar.update(1)

    progress_bar.close()

    metrics = {}
    if accelerator.is_main_process:
        # --- Remove Loss/Perplexity Calculation ---
        # No loss is calculated in generation mode this way
        metrics["loss"] = np.nan
        metrics["perplexity"] = np.nan
        logger.info(f"Base Model Eval Loss/Perplexity: N/A (Generation Mode)")

        if all_preds_decoded and all_labels_decoded:
            logger.info(f"Calculating EM/F1 for {len(all_preds_decoded)} examples...")
            try:
                # Pass the normalized lists directly
                qa_scores = calculate_qa_metrics(all_preds_decoded, all_labels_decoded)
                metrics.update(qa_scores)
                logger.info(f"Base Model Eval EM: {metrics.get('exact_match', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Could not calculate QA metrics (EM/F1): {e}", exc_info=True)
                metrics['exact_match'] = np.nan
                metrics['f1'] = np.nan
        else:
            logger.warning("No predictions or labels were decoded, skipping QA metrics.")
            metrics['exact_match'] = np.nan
            metrics['f1'] = np.nan

    accelerator.wait_for_everyone()
    # Return only metrics calculated on the main process
    return metrics if accelerator.is_main_process else {}

def main():
    global args
    args = parse_arguments()
    config = load_config(args.config)
    base_model_name_short = config['model']['name'].split('/')[-1] # Get short name for logging/wandb

    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")

    # --- Wandb Initialization (only on main process) ---
    wandb_initialized = False
    if accelerator.is_main_process:
        # Default run name if not provided
        run_name = args.wandb_run_name if args.wandb_run_name else f"eval_{base_model_name_short}"
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={**config, **vars(args)} # Log combined config
            )
            wandb_initialized = True
            logger.info(f"Wandb initialized for project '{args.wandb_project}', run '{run_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Wandb: {e}. Reporting disabled.")

    final_metrics = {}
    try:
        # --- Load Base Model and Tokenizer ---
        # No adapter path needed
        model, tokenizer = load_model_and_tokenizer(config)

        # --- Prepare Dataloader ---
        eval_dataloader, eval_dataset = prepare_eval_dataloader(tokenizer, config, args)

        # --- Prepare for Accelerator ---
        model, eval_dataloader_prepared = accelerator.prepare(model, eval_dataloader)

        # --- Run Evaluation ---
        logger.info(f"\n--- Evaluating Base Model: {config['model']['name']} ---")
        metrics = run_evaluation(accelerator, model, tokenizer, eval_dataloader_prepared, eval_dataset)

        if accelerator.is_main_process and metrics:
            final_metrics = metrics # Store metrics from main process
            logger.info(f"Final Base Model Results: {metrics}")

            # --- Log metrics to Wandb ---
            if wandb_initialized:
                # Log metrics directly, maybe prefix with 'eval/'
                wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
                wandb.log(wandb_metrics)
                logger.info(f"Logged base model metrics to Wandb.")

        # --- Clean up ---
        del model
        del tokenizer
        del eval_dataloader_prepared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    except Exception as e:
        logger.error(f"An error occurred during the base model evaluation: {e}", exc_info=True)
    finally:
        if accelerator.is_main_process and wandb_initialized:
            wandb.finish()
            logger.info("Wandb run finished.")

    # Print final results at the end
    if accelerator.is_main_process:
        print("\n--- Base Model Evaluation Summary ---")
        if final_metrics:
             print(f"Model: {config['model']['name']}")
             for metric, value in final_metrics.items():
                 if isinstance(value, (int, float)) and not math.isnan(value):
                     print(f"  {metric}: {value:.4f}")
                 else:
                     print(f"  {metric}: {value}")
             print("-" * 20)
        else:
            print("Base model evaluation did not produce results.")


if __name__ == "__main__":
    main()