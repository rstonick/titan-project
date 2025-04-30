import sys
import os
import argparse # Make sure argparse is imported

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
import yaml
import numpy as np # Ensure numpy is imported

# --- Define compute_metrics function for Trainer ---
def compute_metrics(eval_preds):
    """
    Called by Trainer to compute metrics during evaluation.
    Handles decoding, prefix removal, and metric calculation.
    """
    global global_tokenizer
    if global_tokenizer is None:
        logger.error("Tokenizer not set globally for compute_metrics.")
        return {"exact_match": 0.0, "f1": 0.0}

    preds_ids, label_ids = eval_preds
    logger.info(f"compute_metrics received preds_ids type: {type(preds_ids)}, label_ids type: {type(label_ids)}")

    # --- Robust Conversion Function (Keep for labels) ---
    def convert_to_list_of_int_lists(ids_array, name):
        if isinstance(ids_array, np.ndarray):
            logger.info(f"Converting {name} from np.ndarray to list of lists of int.")
            try:
                # Ensure inner elements are Python ints
                return [[int(token_id) for token_id in seq] for seq in ids_array]
            except Exception as e:
                logger.error(f"Failed during detailed conversion of {name}: {e}")
                return None # Signal error
        elif isinstance(ids_array, (list, tuple)):
             logger.info(f"Converting {name} from {type(ids_array)} to list of lists of int.")
             try:
                  # Ensure inner elements are Python ints
                  return [[int(token_id) for token_id in seq] for seq in ids_array]
             except Exception as e:
                  logger.error(f"Failed during detailed conversion of {name}: {e}")
                  return None # Signal error
        else:
            logger.error(f"Unexpected type for {name}: {type(ids_array)}. Cannot convert.")
            return None # Signal error
    # --- End Conversion Function ---

    preds_ids_list = convert_to_list_of_int_lists(preds_ids, "preds_ids")
    label_ids_list = convert_to_list_of_int_lists(label_ids, "label_ids")

    if preds_ids_list is None or label_ids_list is None:
         return {"exact_match": 0.0, "f1": 0.0, "error": "ID conversion failed"}

    # Replace -100 in labels (use the converted list)
    label_ids_list = [[token_id if token_id != -100 else global_tokenizer.pad_token_id for token_id in seq] for seq in label_ids_list]

    # --- Decode One by One ---
    decoded_preds = []
    decoded_labels = []
    logger.info("Attempting to decode predictions and labels one by one...")
    try:
        # Decode Predictions individually
        for i, seq in enumerate(preds_ids):
            # Convert sequence if it's numpy
            if isinstance(seq, np.ndarray):
                seq = seq.tolist()
            # Ensure elements are ints (belt and braces)
            seq = [int(token_id) for token_id in seq]
            try:
                decoded_preds.append(global_tokenizer.decode(seq, skip_special_tokens=True))
            except Exception as e:
                 logger.error(f"Error decoding prediction sequence {i}: {seq[:20]}... Error: {e}", exc_info=True)
                 decoded_preds.append(f"DECODING_ERROR: {e}") # Add placeholder on error

        # Decode Labels individually (using the converted list)
        for i, seq in enumerate(label_ids_list):
            try:
                decoded_labels.append(global_tokenizer.decode(seq, skip_special_tokens=True))
            except Exception as e:
                 logger.error(f"Error decoding label sequence {i}: {seq[:20]}... Error: {e}", exc_info=True)
                 decoded_labels.append(f"DECODING_ERROR: {e}") # Add placeholder on error

        logger.info("Finished decoding one by one.")

    except Exception as e:
         logger.error(f"General error during one-by-one decoding loop: {e}", exc_info=True)
         return {"exact_match": 0.0, "f1": 0.0, "error": "one-by-one decoding failed"}
    # --- End Decode One by One ---


    # --- Prefix Removal ---
    # The decoded_labels contain the prompt + answer. We need to extract the answer part
    # and also extract the prompt part to remove it from the predictions.
    cleaned_preds = []
    actual_answers = []
    separator = "\nAnswer:" # The separator used in formatting

    for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
        # Skip if decoding failed for either pred or label
        if "DECODING_ERROR" in pred or "DECODING_ERROR" in label:
             logger.warning(f"Skipping metrics for example {i} due to decoding error.")
             continue

        # Find the separator in the label to isolate the prompt and the actual answer
        if separator in label:
            prompt_part = label.split(separator, 1)[0] + separator
            answer_part = label.split(separator, 1)[1].strip()
        else:
            # If separator not found in label (e.g., due to truncation),
            # assume label is mostly answer, and prompt removal might fail.
            logger.warning(f"Separator '{separator}' not found in decoded label: {repr(label)}. Prefix removal might be inaccurate.")
            prompt_part = "" # Cannot reliably determine prompt
            answer_part = label.strip() # Treat whole label as answer

        actual_answers.append(answer_part)

        # Remove the prompt_part from the prediction
        # Use strip() on prompt_part to handle potential leading/trailing spaces from decoding
        clean_prompt = prompt_part.strip()
        processed_pred = pred # Default to full prediction

        if clean_prompt and pred.strip().startswith(clean_prompt):
             # Remove prefix if found (use strip on pred for matching)
             processed_pred = pred.strip()[len(clean_prompt):].strip()
        elif clean_prompt:
             # Log mismatch if prompt was expected but not found
             logger.warning(f"Prefix mismatch during metric calculation. Prompt: {repr(clean_prompt)}, Pred: {repr(pred.strip()[:len(clean_prompt)+20])}...")
             # Keep full prediction as fallback
             processed_pred = pred.strip()
        else:
             # If prompt_part was empty, keep full prediction
             processed_pred = pred.strip()

        cleaned_preds.append(processed_pred)

    # Calculate metrics using the cleaned predictions and extracted answers
    if not cleaned_preds: # Handle case where all examples had decoding errors
         logger.error("No examples could be processed for metrics due to decoding errors.")
         return {"exact_match": 0.0, "f1": 0.0, "error": "All examples failed decoding"}

    metrics = calculate_qa_metrics(cleaned_preds, actual_answers)
    return metrics


# --- Main Evaluation Function (Refactored for Trainer) ---
def run_evaluation(accelerator, model, tokenizer, eval_dataset, adapter_name, config):
    """Runs evaluation using the Hugging Face Trainer."""
    global global_tokenizer # Set the global tokenizer
    global_tokenizer = tokenizer

    logger.info(f"--- Evaluating Model: {adapter_name} using Trainer ---")

    # 1. Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 2. Training Arguments (for evaluation)
    output_dir = f"./eval_temp_{adapter_name}" # Temporary directory
    try:
        eval_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=config['evaluation']['batch_size'],
            dataloader_num_workers=config['evaluation']['num_workers'],
            remove_unused_columns=False,
            eval_accumulation_steps=config['evaluation'].get('eval_accumulation_steps', None),
            fp16=accelerator.mixed_precision == 'fp16', # Use accelerator.mixed_precision
            # bf16=accelerator.mixed_precision == 'bf16', # Use accelerator.mixed_precision (Commented out if bf16 not used)
            report_to=["wandb"] if config['wandb']['log_eval'] else ["none"],
            logging_dir=f"{output_dir}/logs",
        )
    except KeyError as e:
         logger.error(f"[{adapter_name}] Configuration missing key required for TrainingArguments: {e}. Cannot proceed with evaluation.")
         return {'loss': float('nan'), 'perplexity': float('nan'), 'exact_match': 0.0, 'f1': 0.0, 'error': f'Missing config key: {e}'}

    # --- REMOVE checks on eval_args.generation_config ---
    # if eval_args.generation_config.pad_token_id is None:
    #     eval_args.generation_config.pad_token_id = tokenizer.pad_token_id
    # if eval_args.generation_config.eos_token_id is None:
    #     eval_args.generation_config.eos_token_id = tokenizer.eos_token_id
    # --- End REMOVE ---

    # 3. Instantiate Trainer
    trainer = Trainer(
        model=model, # Model already has generation_config set in main
        args=eval_args,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer, # Removed: Trainer can infer from model/collator
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # label_names=["labels"] # Removed: Causes TypeError in this version
    )

    # 4. Run Evaluation
    logger.info(f"[{adapter_name}] Starting evaluation with Trainer...")
    try:
        eval_results = trainer.evaluate()
        # Metrics are computed by compute_metrics and included in eval_results
        # Example: eval_results might be {'eval_loss': ..., 'eval_exact_match': ..., 'eval_f1': ..., 'eval_runtime': ...}
        logger.info(f"[{adapter_name}] Trainer evaluation finished.")

        # Extract the metrics we care about (remove 'eval_' prefix added by Trainer)
        metrics = {k.replace("eval_", ""): v for k, v in eval_results.items() if k.startswith("eval_")}
        # Add loss/perplexity if available (might be NaN if only generation)
        metrics['loss'] = eval_results.get('eval_loss', float('nan'))
        metrics['perplexity'] = np.exp(metrics['loss']) if not np.isnan(metrics['loss']) else float('nan')

    except Exception as e:
        logger.error(f"[{adapter_name}] Error during Trainer evaluation: {e}", exc_info=True)
        metrics = {'loss': float('nan'), 'perplexity': float('nan'), 'exact_match': 0.0, 'f1': 0.0}

    # Clean up temporary directory? (Optional)
    # import shutil
    # shutil.rmtree(output_dir)

    return metrics


# --- Main Execution Logic ---
def main():
    # Ensure parse_arguments() is called correctly
    args = parse_arguments()
    config = load_config(args.config)

    # --- Accelerator Initialization ---
    # No project_config needed unless saving state during evaluation
    accelerator = Accelerator()
    logger.info(f"Accelerator initialized. Device: {accelerator.device}, Num processes: {accelerator.num_processes}")

    # --- Wandb Initialization (only on main process) ---
    wandb_initialized = False
    # Check if wandb logging is enabled in config
    if config.get('wandb', {}).get('log_eval', False) and accelerator.is_main_process:
        try:
            wandb.init(
                project=config['wandb']['project'], # Use config value
                entity=config['wandb'].get('entity'), # Use config value (optional)
                # name=args.wandb_run_name, # Name is usually auto-generated or set in config
                config=config # Log the loaded config
            )
            wandb_initialized = True
            logger.info(f"Wandb initialized for project '{config['wandb']['project']}'")
        except KeyError as e:
             logger.error(f"Wandb configuration missing key: {e}. Reporting disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize Wandb: {e}. Reporting disabled.")

    # --- Prepare Tokenizer ---
    try:
        # Correct the import statement
        from code.data_preparation import prepare_tokenizer
        tokenizer = prepare_tokenizer(config)
    except ImportError:
        logger.error("Could not import prepare_tokenizer from code.data_preparation.py. Check path and file existence.")
        sys.exit(1)
    except NameError:
        logger.error("prepare_tokenizer function not found after import. Check data_preparation.py.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to prepare tokenizer: {e}", exc_info=True)
        sys.exit(1)


    # --- Prepare Dataset ---
    try:
        # Correct the import statement
        from code.data_preparation import prepare_dataset
        tokenized_datasets = prepare_dataset(tokenizer, config)
        eval_dataset = tokenized_datasets['validation']
        logger.info(f"Using {len(eval_dataset)} examples for evaluation.")
        if len(eval_dataset) == 0:
             logger.error("Evaluation dataset is empty. Exiting.")
             return # Exit if dataset is empty
    except ImportError:
         logger.error("Could not import prepare_dataset from code.data_preparation.py. Check path and file existence.")
         sys.exit(1)
    except NameError:
         logger.error("prepare_dataset function not found after import. Check data_preparation.py.")
         sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}", exc_info=True)
        return

    # --- Load Models and Evaluate ---
    # Use output_dir from config
    models_parent_dir = config['model']['output_dir']
    try:
        adapter_dirs = [d for d in os.listdir(models_parent_dir)
                        if os.path.isdir(os.path.join(models_parent_dir, d))]
        adapter_paths = [os.path.join(models_parent_dir, d) for d in adapter_dirs]
    except FileNotFoundError:
        logger.error(f"Model output directory not found: {models_parent_dir}")
        return
    except Exception as e:
        logger.error(f"Error listing adapter directories in {models_parent_dir}: {e}")
        return

    all_results = {}

    if not adapter_paths:
        logger.warning(f"No adapter directories found in {models_parent_dir}. Nothing to evaluate.")
        return

    # --- Define BitsAndBytesConfig ---
    # Ensure bnb_config is defined using values from config
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization']['load_in_4bit'],
            bnb_4bit_use_double_quant=config['quantization']['use_double_quant'],
            bnb_4bit_quant_type=config['quantization']['quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype']) # Get torch dtype from string
        )
    except KeyError as e:
        logger.error(f"Quantization configuration missing key: {e}. Cannot proceed.")
        return
    except AttributeError:
         logger.error(f"Invalid torch dtype specified in quantization config: {config['quantization']['compute_dtype']}. Cannot proceed.")
         return
    except Exception as e:
         logger.error(f"Error creating BitsAndBytesConfig: {e}. Cannot proceed.")
         return


    for adapter_path in adapter_paths:
        adapter_name = os.path.basename(adapter_path)
        logger.info(f"\n--- Loading Model: {adapter_name} ---")
        # --- Load Base Model ---
        try:
            # Determine torch dtype based on accelerator precision or config
            # Use accelerator.mixed_precision
            compute_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
            if 'compute_dtype' in config['quantization']:
                 try:
                      compute_dtype = getattr(torch, config['quantization']['compute_dtype'])
                 except AttributeError:
                      logger.warning(f"Invalid compute_dtype '{config['quantization']['compute_dtype']}' in config, falling back.")

            base_model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                quantization_config=bnb_config,
                device_map={"": accelerator.process_index},
                trust_remote_code=True,
                torch_dtype=compute_dtype, # Use determined compute_dtype
            )
            logger.info(f"Loaded base model {config['model']['name']}.")

            # --- Load LoRA Adapter ---
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            try:
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.eval()
                # logger.info(f"Successfully loaded and merged LoRA adapter onto the base model from {adapter_path}") # <-- Adjust log message
                logger.info(f"Successfully loaded LoRA adapter onto the base model from {adapter_path}")


                # --- Set AND CHECK Generation Config ---
                # Ensure pad_token_id is set correctly ON THE MODEL (PeftModel handles this)
                if model.generation_config.pad_token_id is None:
                     model.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                     logger.info(f"[{adapter_name}] Set model.generation_config.pad_token_id to: {model.generation_config.pad_token_id}")
                if model.generation_config.eos_token_id is None:
                     model.generation_config.eos_token_id = tokenizer.eos_token_id
                     logger.info(f"[{adapter_name}] Set model.generation_config.eos_token_id to: {model.generation_config.eos_token_id}")

                # Set max_new_tokens for generation from config
                try:
                    model.generation_config.max_new_tokens = config['evaluation']['max_new_tokens']
                except KeyError:
                     logger.error("Missing 'evaluation.max_new_tokens' in config. Cannot set generation length.")
                     all_results[adapter_name] = {'error': 'Missing evaluation.max_new_tokens in config'}
                     # Clean up before continuing
                     del model
                     del base_model
                     if torch.cuda.is_available():
                          torch.cuda.empty_cache()
                     continue # Skip to next adapter

                logger.info(f"[{adapter_name}] Final Generation config: {model.generation_config}")

                # --- Run Evaluation using Trainer (with PeftModel) ---
                metrics = run_evaluation(accelerator, model, tokenizer, eval_dataset, adapter_name, config)
                all_results[adapter_name] = metrics

                # --- Log Results ---
                # Check wandb_initialized flag before logging
                if wandb_initialized and accelerator.is_main_process:
                    # Check if metrics dict contains an error key before logging
                    if 'error' not in metrics:
                        wandb_metrics = {f"eval/{adapter_name}/{k}": v for k, v in metrics.items()}
                        wandb.log(wandb_metrics)
                        logger.info(f"Logged metrics for {adapter_name} to Wandb.")
                    else:
                        logger.warning(f"Skipping Wandb logging for {adapter_name} due to evaluation error: {metrics['error']}")

                # --- Optional: Merge after evaluation if needed elsewhere ---
                # model = model.merge_and_unload()
                # logger.info(f"[{adapter_name}] Merged adapter after evaluation.")


            except Exception as e:
                logger.error(f"Failed to load or evaluate adapter {adapter_name}: {e}", exc_info=True)
                all_results[adapter_name] = {'error': str(e)}
            finally:
                 # Clean up model resources if possible (important with large models)
                 # Ensure both model (PeftModel) and base_model are deleted
                 if 'model' in locals(): del model
                 if 'base_model' in locals(): del base_model
                 if torch.cuda.is_available():
                      torch.cuda.empty_cache()


        except Exception as e:
            logger.error(f"Failed to load base model {config['model']['name']}: {e}", exc_info=True)
            # Cannot proceed if base model fails
            break


    # --- Final Summary ---
    logger.info("\n--- Evaluation Summary ---")
    for adapter_name, results in all_results.items():
        logger.info(f"{adapter_name}: {results}")

    if not all_results:
         logger.info("No models were successfully evaluated.")

    # Check wandb_initialized flag before finishing
    if wandb_initialized and accelerator.is_main_process:
        wandb.finish()
        logger.info("Wandb run finished.")

if __name__ == "__main__":
    # Imports are now handled within main() after modifying sys.path
    main()