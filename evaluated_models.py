import argparse
import os
import torch
import yaml
import math
import numpy as np
from tqdm.auto import tqdm
import logging # Use standard logging

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel # Import PeftModel for loading adapters
from accelerate import Accelerator
from accelerate.logging import get_logger # Use accelerator logger if preferred

# Import necessary functions from your project structure
from code.config_loader import load_config
from code.data_preparation import prepare_tokenizer, prepare_dataset # Reuse dataset prep logic
from code.metrics import calculate_qa_metrics # Reuse metrics calculation

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use standard Python logger

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral LoRA models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file used during training.')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing the saved model checkpoints (e.g., final_outputs/). Each subdirectory should contain a model.')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='Override evaluation batch size per device. Defaults to value in config.')
    return parser.parse_args()

def load_model_and_tokenizer(model_path, config):
    """Load the base model, apply LoRA adapter, and load the tokenizer."""
    logger.info(f"Loading base model: {config['model']['name']}")
    
    # --- Load Tokenizer ---
    # Use the same tokenizer preparation as in training
    tokenizer = prepare_tokenizer(config)
    logger.info(f"Loaded tokenizer from {config['model']['name']}")

    # --- Load Base Model (with quantization if used during training) ---
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
        config['model']['name'],
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, config['model'].get('torch_dtype', "float16")), # Match training dtype
        device_map="auto", # Let Accelerate handle device placement later if possible, or use auto for basic loading
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    logger.info(f"Loaded base model {config['model']['name']}.")

    # --- Load LoRA Adapter ---
    logger.info(f"Loading LoRA adapter from: {model_path}")
    # Ensure the path exists
    if not os.path.isdir(model_path):
         raise FileNotFoundError(f"Adapter directory not found: {model_path}")
         
    # Check for adapter files before loading
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    adapter_model_path = os.path.join(model_path, "adapter_model.bin") # Or .safetensors
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {model_path}")
    # Add check for adapter_model.bin or adapter_model.safetensors if needed

    # Load the PEFT model by applying the adapter to the base model
    model = PeftModel.from_pretrained(model, model_path)
    logger.info(f"Successfully loaded LoRA adapter onto the base model from {model_path}")

    return model, tokenizer

def prepare_eval_dataloader(tokenizer, config):
    """Prepare the evaluation dataloader using the validation split."""
    logger.info("Preparing evaluation dataset...")
    # Reuse the dataset preparation logic, but we only need the validation split
    # Note: This will reload the dataset. Consider optimizing if evaluating many models.
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
        shuffle=False # No need to shuffle for evaluation
    )
    return eval_dataloader, eval_dataset # Return dataset for length calculation

def run_evaluation(accelerator, model, tokenizer, eval_dataloader, eval_dataset):
    """Runs the evaluation loop and calculates metrics."""
    model.eval()
    losses = []
    all_preds_decoded = []
    all_labels_decoded = []

    progress_bar = tqdm(total=len(eval_dataloader), desc="Evaluating", disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        # Ensure loss is gathered correctly even if it's already on one device
        gathered_loss = accelerator.gather_for_metrics(loss.reshape(1, -1)) # Reshape to handle single value loss
        losses.append(gathered_loss)

        # Generate predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        # Gather preds and labels across devices
        gathered_preds = accelerator.gather_for_metrics(preds)
        gathered_labels = accelerator.gather_for_metrics(labels)

        # Decode tokens on the main process after gathering
        if accelerator.is_main_process:
            # Ensure tensors are on CPU and are numpy arrays for decoding
            gathered_preds_np = gathered_preds.cpu().numpy()
            gathered_labels_np = gathered_labels.cpu().numpy()

            # Replace -100 with pad_token_id before decoding
            label_pad_token_id = -100
            gathered_labels_np[gathered_labels_np == label_pad_token_id] = tokenizer.pad_token_id
            gathered_preds_np[gathered_preds_np == label_pad_token_id] = tokenizer.pad_token_id # Should not happen with argmax, but safe

            # Decode, skipping special tokens
            decoded_preds_batch = tokenizer.batch_decode(gathered_preds_np, skip_special_tokens=True)
            decoded_labels_batch = tokenizer.batch_decode(gathered_labels_np, skip_special_tokens=True)

            # Clean up whitespace and store
            all_preds_decoded.extend([p.strip() for p in decoded_preds_batch])
            all_labels_decoded.extend([l.strip() for l in decoded_labels_batch])

        progress_bar.update(1)

    progress_bar.close()

    # Calculate final metrics on the main process
    metrics = {}
    if accelerator.is_main_process:
        # Calculate Loss and Perplexity
        losses = torch.cat(losses)
        try:
            # Truncate losses if gathered more samples than dataset size (due to batch padding)
            if len(losses) > len(eval_dataset):
                losses = losses[:len(eval_dataset)]
            eval_loss = torch.mean(losses).item() # Get scalar value
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
            eval_loss = float("inf")
        except Exception as e:
             logger.error(f"Error calculating eval loss/perplexity: {e}")
             eval_loss = float("nan")
             perplexity = float("nan")

        metrics["loss"] = eval_loss
        metrics["perplexity"] = perplexity
        logger.info(f"Evaluation Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")

        # Calculate QA Metrics (EM, F1)
        if all_preds_decoded and all_labels_decoded:
            try:
                # Assuming labels are single strings after decoding based on training script
                # If your dataset has multiple reference answers, adjust calculate_qa_metrics or data prep
                qa_scores = calculate_qa_metrics(all_preds_decoded, all_labels_decoded)
                metrics.update(qa_scores) # Adds 'exact_match' and 'f1'
                logger.info(f"Evaluation EM: {metrics.get('exact_match', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Could not calculate QA metrics (EM/F1): {e}")
                metrics['exact_match'] = np.nan
                metrics['f1'] = np.nan
        else:
            logger.warning("No predictions or labels were decoded, skipping QA metrics.")

    # Synchronize processes before returning metrics from the main process
    accelerator.wait_for_everyone()

    # Return metrics dictionary (only main process has the full dict)
    return metrics if accelerator.is_main_process else {}

def main():
    global args # Make args accessible globally within main() scope if needed elsewhere
    args = parse_arguments()
    config = load_config(args.config)

    # --- Accelerator Initialization ---
    # No project_config needed unless saving state during evaluation
    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")

    # --- Prepare Dataloader (once) ---
    # Need a temporary tokenizer to prepare the dataset first
    temp_tokenizer = prepare_tokenizer(config)
    eval_dataloader, eval_dataset = prepare_eval_dataloader(temp_tokenizer, config)
    del temp_tokenizer # Free memory

    # --- Find Model Directories ---
    if not os.path.isdir(args.models_dir):
        logger.error(f"Models directory not found: {args.models_dir}")
        return

    model_dirs = [
        os.path.join(args.models_dir, d)
        for d in os.listdir(args.models_dir)
        if os.path.isdir(os.path.join(args.models_dir, d))
    ]

    if not model_dirs:
        logger.error(f"No subdirectories found in {args.models_dir}. Expecting directories containing saved models.")
        return

    logger.info(f"Found {len(model_dirs)} potential model directories to evaluate.")

    # --- Evaluation Loop ---
    all_results = {}
    for model_dir in model_dirs:
        logger.info(f"\n--- Evaluating Model: {model_dir} ---")
        try:
            # --- Load Model and Tokenizer for the current directory ---
            model, tokenizer = load_model_and_tokenizer(model_dir, config)

            # --- Prepare for Accelerator ---
            # Accelerator prepares the model and dataloader for distributed execution
            model, eval_dataloader_prepared = accelerator.prepare(model, eval_dataloader)
            
            # --- Run Evaluation ---
            metrics = run_evaluation(accelerator, model, tokenizer, eval_dataloader_prepared, eval_dataset)

            if accelerator.is_main_process:
                 all_results[model_dir] = metrics
                 logger.info(f"Results for {model_dir}: {metrics}")

            # --- Clean up ---
            # Explicitly delete model and clear cache to free GPU memory before loading the next one
            del model
            del tokenizer
            del eval_dataloader_prepared # Delete the prepared dataloader too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone() # Ensure cleanup happens everywhere

        except FileNotFoundError as e:
             logger.error(f"Skipping {model_dir}: {e}")
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_dir}: {e}", exc_info=True) # Log traceback
            # Clean up even on error
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            if 'eval_dataloader_prepared' in locals(): del eval_dataloader_prepared
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()


    # --- Print Summary ---
    if accelerator.is_main_process:
        logger.info("\n--- Evaluation Summary ---")
        if all_results:
            for model_path, results in all_results.items():
                print(f"Model: {model_path}")
                for metric, value in results.items():
                    print(f"  {metric}: {value:.4f}")
                print("-" * 20)
        else:
            print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()