import argparse
import os
import torch
import yaml
import wandb
import math
import numpy as np
from tqdm.auto import tqdm
import importlib.metadata
import bitsandbytes as bnb
import shutil # Moved import shutil to the top

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

# Import the specific metric functions needed
from code.metrics_QA import exact_match_score, f1_score, metric_max_over_ground_truths # NEW import for QA
from code.metrics import calculate_perplexity # Keep perplexity if needed

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from code.config_loader import load_config
# from code.data_preparation import prepare_tokenizer, prepare_dataset # OLD import
from code.data_preparation import prepare_tokenizer # Keep tokenizer prep if it's separate
from code.prepare_dataset_local import prepare_local_triviaqa_rc_nocontext # NEW import for local data prep
from code.model_preparation import prepare_model

logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train Mistral LoRA model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    return parser.parse_args()


def main():
    # --- Configuration Loading ---
    args = parse_arguments()
    config = load_config(args.config)

    # --- Accelerator Initialization ---
    # Add project_config for checkpoint management
    output_dir = config['model']['output_dir']
    project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=os.path.join(output_dir, "logs"))

    # Initialize accelerator with gradient accumulation and logging
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        log_with="wandb",
        project_config=project_config # Added project_config
    )
    # --- End Accelerator Initialization ---

    # --- Seed and Logging ---
    # Make one log on every process with the configuration for debugging.
    logger.info(accelerator.state, main_process_only=False)

    # --- WandB Initialization ---
    # Init wandb (handled by accelerator)
    # Check if main process for initialization
    if accelerator.is_main_process:
        wandb_config = config.get('wandb', {})
        run_name = config.get('model', {}).get('wandb_run_name', 'default_run_name')

        # --- Log Environment Info ---
        try:
            env_info = {
                "torch_version": importlib.metadata.version("torch"),
                "transformers_version": importlib.metadata.version("transformers"),
                "accelerate_version": importlib.metadata.version("accelerate"),
                "peft_version": importlib.metadata.version("peft"),
                "datasets_version": importlib.metadata.version("datasets"),
                # Add other relevant libraries
            }
            # Merge env_info into the main config for logging
            config['environment'] = env_info
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not log library versions - some packages might not be installed.")
        # --- End Log Environment Info ---


        # Initialize Wandb run through accelerator
        accelerator.init_trackers(
            project_name=wandb_config.get('project', 'mistral-lora-finetuning-accelerate'),
            config=config, # config now includes environment info
            init_kwargs={"wandb": {"name": run_name}}
        )


    # --- Prepare Model and Tokenizer ---
    model = prepare_model(config)
    tokenizer = prepare_tokenizer(config)

    # --- Prepare Dataset and Dataloaders using local script ---
    # tokenized_dataset = prepare_dataset(tokenizer, config) # OLD call
    tokenized_dataset = prepare_local_triviaqa_rc_nocontext(tokenizer, config) # NEW call

    # Split the dataset into train and eval
    if 'validation' not in tokenized_dataset or not tokenized_dataset['validation']:
         # If no validation split exists or is empty, create one (e.g., 10% of train)
         logger.warning("No validation split found or it is empty. Creating one from 10% of the training data.")
         # Ensure train split is not empty before splitting
         if not tokenized_dataset['train']:
              raise ValueError("Training dataset is empty, cannot create validation split.")
         split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=config.get('seed', 42))
         train_dataset = split_dataset['train']
         eval_dataset = split_dataset['test']
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']


    # --- DataLoader Setup ---
    # Ensure the collator doesn't remove the 'normalized_aliases' column
    # DataCollatorForLanguageModeling usually only uses input_ids, attention_mask, labels
    # If using a custom collator, make sure it passes 'normalized_aliases' through.
    # Standard collator should be fine as it doesn't explicitly remove columns it doesn't use.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Read num_workers from config, default to 0 if not specified
    num_workers = config['trainer'].get('dataloader_num_workers', 0)
    logger.info(f"Using {num_workers} dataloader workers.") # Log the value being used

    # --- Modification Start: Remove problematic columns before DataLoader ---
    # Keep the original eval_dataset for metrics calculation later
    original_eval_dataset_for_metrics = eval_dataset

    # Define columns to remove (those not needed by the model/collator)
    columns_to_remove = []
    if 'normalized_aliases' in train_dataset.column_names:
        columns_to_remove.append('normalized_aliases')
        # Add any other columns here that cause issues or are unnecessary for training/model input
        # e.g., columns_to_remove.extend(['question_id', 'entity_pages'])

    # Create copies for DataLoaders with columns removed
    if columns_to_remove:
        logger.info(f"Removing columns before creating DataLoaders: {columns_to_remove}")
        train_dataloader_dataset = train_dataset.remove_columns(columns_to_remove)
        eval_dataloader_dataset = eval_dataset.remove_columns(columns_to_remove)
    else:
        logger.info("No columns specified for removal before creating DataLoaders.")
        train_dataloader_dataset = train_dataset
        eval_dataloader_dataset = eval_dataset
    # --- Modification End ---


    train_dataloader = DataLoader(
        train_dataloader_dataset, # Use the dataset with columns removed
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config['training']['batch_size'], # Per device batch size
        num_workers=num_workers
    )

    # Eval dataloader (Uncommented and configured)
    eval_batch_size = config['trainer'].get('per_device_eval_batch_size', config['training']['batch_size'])
    eval_dataloader = DataLoader(
        eval_dataloader_dataset, # Use the dataset with columns removed
        collate_fn=data_collator,
        batch_size=eval_batch_size, # Use configured eval batch size
        num_workers=num_workers
    )
    # --- End DataLoader Setup ---


    # --- Optimizer and Scheduler ---
    # Use PagedAdamW8bit from bitsandbytes
    # Explicitly cast learning_rate to float
    learning_rate = float(config['training']['learning_rate'])
    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(), # Ensure optimizer gets parameters from the PEFT model
        lr=learning_rate, # Use the float variable
        weight_decay=config['trainer'].get('weight_decay', 0.0),
        # Add other PagedAdamW8bit specific arguments if needed (e.g., optim_bits=8)
        # Check bitsandbytes documentation for available arguments
    )

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['training']['gradient_accumulation_steps'])
    max_train_steps = config['training']['num_epochs'] * num_update_steps_per_epoch

    # Create LR scheduler (FIRST AND ONLY TIME)
    lr_scheduler = get_scheduler(
        name=config['trainer'].get('lr_scheduler_type', 'linear'),
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config['training']['warmup_ratio']),
        num_training_steps=max_train_steps,
    )
    # --- End Optimizer and Scheduler Setup ---


    # --- Accelerator Prepare ---
    # Prepare eval_dataloader as well
    # IMPORTANT: eval_dataset is NOT prepared by accelerator, only dataloaders are.
    # We need the original eval_dataset later for accessing 'normalized_aliases'.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # --- End Accelerator Prepare ---

    # --- Wandb Watch ---
    # Call watch on the main process after the model is prepared by accelerator
    if accelerator.is_main_process:
         # log='gradients' or log='all' will log gradients, log_freq controls frequency
         wandb.watch(model, log="gradients", log_freq=config['trainer'].get('logging_steps', 100))
    # --- End Wandb Watch ---


    # --- Training Loop Setup ---
    total_train_batch_size = config['training']['batch_size'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']
    total_eval_batch_size = config['trainer'].get('per_device_eval_batch_size', config['training']['batch_size']) * accelerator.num_processes # Use eval batch size if specified

    # --- Define evaluation and saving strategies BEFORE the loop ---
    evaluation_strategy = config['trainer'].get('evaluation_strategy', 'no') # Default to 'no' if not specified
    save_strategy = config['trainer'].get('save_strategy', 'no') # Default to 'no' if not specified
    eval_steps = config['trainer'].get('eval_steps')
    save_steps = config['trainer'].get('save_steps')
    logging_steps = config['trainer'].get('logging_steps', 10) # Default logging steps

    # --- Early Stopping Setup ---
    early_stopping_enabled = config['trainer'].get('load_best_model_at_end', False)
    metric_for_best_model = config['trainer'].get('metric_for_best_model', 'loss') # Default to loss
    greater_is_better = config['trainer'].get('greater_is_better', False) # Default to False for loss
    best_metric_value = float('inf') if not greater_is_better else float('-inf')
    best_checkpoint_path = None
    save_total_limit = config['trainer'].get('save_total_limit') # Get save limit
    saved_checkpoints = [] # Track saved checkpoints for limit

    # Function to compare metrics based on greater_is_better
    def is_better(current_metric, best_metric):
        if greater_is_better:
            return current_metric > best_metric
        else:
            return current_metric < best_metric

    # --- Progress Bar ---
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    global_step = 0 # Track total steps for logging/saving/eval
    starting_epoch = 0

    # --- Checkpoint Resuming ---
    resume_step = 0 # Initialize resume_step
    if project_config.automatic_checkpoint_naming and os.path.isdir(output_dir):
        checkpoints = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ], key=lambda x: int(x.split('-')[-1]))
        if checkpoints:
            resume_from_checkpoint = os.path.join(output_dir, checkpoints[-1])
            logger.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            try:
                global_step = int(resume_from_checkpoint.split("-")[-1])
                starting_epoch = global_step // num_update_steps_per_epoch
                resume_step = global_step % num_update_steps_per_epoch # Calculate steps completed in current epoch
                logger.info(f"Resumed at step {global_step}, epoch {starting_epoch}, resume_step {resume_step}")
                progress_bar.update(global_step)
                # NOTE: Relying on accelerator.load_state to handle dataloader state.
                # If manual skipping is needed, use accelerator.skip_first_batches below.
            except ValueError:
                logger.error(f"Could not parse step number from checkpoint path {resume_from_checkpoint}. Starting from scratch.")
                global_step = 0
                starting_epoch = 0
                resume_step = 0


    # --- Evaluation Function ---
    # Pass tokenizer as an argument
    def evaluate(eval_tokenizer):
        model.eval()
        losses = []
        all_preds_decoded = []
        # No longer need all_labels_decoded for ground truths
        # all_labels_decoded = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            # Ensure loss is gathered correctly even if batch size is 1
            gathered_loss = accelerator.gather_for_metrics(loss.reshape(-1))
            losses.append(gathered_loss)

            # --- Add prediction generation and gathering ---
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            # labels = batch["labels"] # Only needed if calculating loss manually here

            # Gather preds across devices
            gathered_preds = accelerator.gather_for_metrics(preds)
            # gathered_labels = accelerator.gather_for_metrics(labels)

            # Decode tokens on the main process after gathering
            if accelerator.is_main_process:
                gathered_preds_np = gathered_preds.cpu().numpy()
                gathered_preds_np[gathered_preds_np == -100] = eval_tokenizer.pad_token_id
                decoded_preds_batch = eval_tokenizer.batch_decode(gathered_preds_np, skip_special_tokens=True)
                all_preds_decoded.extend([p.strip() for p in decoded_preds_batch])

            # Note: Ground truths (normalized_aliases) are handled post-loop
            # --- End prediction generation and gathering ---

        # --- Post-loop processing on main process ---
        metrics = {}
        if accelerator.is_main_process:
            losses = torch.cat(losses)
            try:
                # Truncate losses to the size of the original eval_dataset
                # Use the unprepared eval_dataset length
                if len(losses) > len(eval_dataset):
                     logger.info(f"Truncating gathered losses ({len(losses)}) to eval_dataset size ({len(eval_dataset)})")
                     losses = losses[:len(eval_dataset)]
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
                eval_loss = torch.tensor(float("inf")) # Match type
            except Exception as e:
                 logger.error(f"Error calculating eval loss/perplexity: {e}")
                 eval_loss = torch.tensor(float("nan"))
                 perplexity = float("nan")

            logger.info(f"Evaluation Results - Step: {global_step}, Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
            metrics["loss"] = eval_loss.item() if torch.is_tensor(eval_loss) else eval_loss
            metrics["perplexity"] = perplexity

            # --- Get all ground truths from the original eval_dataset ---
            all_ground_truths = []
            # Use the original dataset that still has the column
            # Ensure we use the variable holding the original dataset
            if 'normalized_aliases' in original_eval_dataset_for_metrics.column_names:
                 # This assumes the order in eval_dataset matches the order of predictions/losses
                 # This is generally true if shuffle=False for eval_dataloader (which is default)
                 all_ground_truths = original_eval_dataset_for_metrics['normalized_aliases']
                 # Truncate ground truths to match potentially truncated predictions/losses
                 # Use the length of the original dataset for comparison if needed, but usually match preds
                 eval_dataset_len = len(original_eval_dataset_for_metrics)
                 if len(losses) > eval_dataset_len:
                     logger.info(f"Truncating gathered losses ({len(losses)}) to original eval_dataset size ({eval_dataset_len})")
                     losses = losses[:eval_dataset_len]

                 if len(all_preds_decoded) < len(all_ground_truths):
                      logger.info(f"Truncating ground truths ({len(all_ground_truths)}) to match predictions ({len(all_preds_decoded)})")
                      all_ground_truths = all_ground_truths[:len(all_preds_decoded)]
                 elif len(all_preds_decoded) > len(all_ground_truths):
                      # This case is less likely but possible if prediction decoding failed for some items
                      logger.warning(f"More predictions ({len(all_preds_decoded)}) than ground truths ({len(all_ground_truths)}). Truncating predictions.")
                      all_preds_decoded = all_preds_decoded[:len(all_ground_truths)]

            else:
                 logger.error("'normalized_aliases' column not found in original_eval_dataset_for_metrics. Cannot calculate QA metrics.")

            # --- Calculate QA Metrics (EM, F1) using all_ground_truths ---
            if all_preds_decoded and all_ground_truths and len(all_preds_decoded) == len(all_ground_truths):
                total_em = 0.0
                total_f1 = 0.0
                num_examples = len(all_preds_decoded)

                for i in range(num_examples):
                    pred = all_preds_decoded[i]
                    ground_truths = all_ground_truths[i] # Directly use the list of aliases

                    # Ensure ground_truths is a list (it should be from the dataset)
                    if not isinstance(ground_truths, list):
                        logger.warning(f"Expected list for ground_truths at index {i}, got {type(ground_truths)}. Skipping.")
                        continue

                    if ground_truths: # Only calculate if list is not empty
                        total_em += metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
                        total_f1 += metric_max_over_ground_truths(f1_score, pred, ground_truths)

                avg_em = total_em / num_examples if num_examples > 0 else 0.0
                avg_f1 = total_f1 / num_examples if num_examples > 0 else 0.0

                metrics["exact_match"] = avg_em
                metrics["f1"] = avg_f1
                logger.info(f"Evaluation EM: {metrics.get('exact_match', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")

            else: # If lists are empty or lengths mismatch
                metrics['exact_match'] = np.nan
                metrics['f1'] = np.nan
                if not all_ground_truths:
                     logger.warning("Could not calculate QA metrics because ground truths list was empty or not found.")
                elif len(all_preds_decoded) != len(all_ground_truths):
                     logger.warning(f"Mismatch between predictions ({len(all_preds_decoded)}) and ground truths ({len(all_ground_truths)}). Cannot calculate QA metrics.")
                else:
                     logger.warning("Could not calculate QA metrics because decoded predictions were empty.")
            # --- End QA Metrics Calculation ---

        # Synchronize processes before returning metrics from the main process
        accelerator.wait_for_everyone()

        model.train() # Set back to train mode

        # Return metrics dictionary (only main process has the full dict)
        return metrics if accelerator.is_main_process else {} # Return empty dict on other processes
    # --- End Evaluation Function ---


    for epoch in range(starting_epoch, config['training']['num_epochs']):
        model.train()
        total_loss = 0
        active_dataloader = train_dataloader # Use the prepared dataloader
        epoch_start_step = global_step # Track step at the start of the epoch for epoch-based eval/save

        # --- Optional: Skip batches if resuming and accelerator doesn't handle it ---
        # This assumes load_state did NOT fully restore the dataloader's position.
        # Only skip if resuming *within* an epoch (resume_step > 0).
        # if resume_step > 0 and epoch == starting_epoch:
        #    logger.info(f"Skipping first {resume_step * config['training']['gradient_accumulation_steps']} batches due to resuming.")
        #    # Adjust num_batches based on gradient accumulation
        #    num_batches_to_skip = resume_step * config['training']['gradient_accumulation_steps']
        #    active_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=num_batches_to_skip)
        #    resume_step = 0 # Reset resume_step after skipping

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()

                accelerator.backward(loss)

                # Optimizer step happens only when gradients are synced
                if accelerator.sync_gradients:
                    # Gradient clipping (optional, add config if needed)
                    # if config['training'].get('max_grad_norm'):
                    #     accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                completed_steps = global_step # Use global_step for clarity

                # --- Logging ---
                if global_step % logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / logging_steps / config['training']['gradient_accumulation_steps']
                    current_lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch}, Step {global_step}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")
                    accelerator.log(
                        {
                            "train_loss": avg_loss,
                            "learning_rate": current_lr,
                            "epoch": epoch + (step + 1) / len(train_dataloader) # More precise epoch tracking
                        },
                        step=global_step,
                    )
                    total_loss = 0 # Reset loss accumulator

                # --- Evaluation and Saving Logic ---
                perform_eval = False
                perform_save = False

                if evaluation_strategy == "steps" and eval_steps and global_step % eval_steps == 0:
                    perform_eval = True
                if save_strategy == "steps" and save_steps and global_step % save_steps == 0:
                    perform_save = True

                # --- Evaluation Trigger ---
                if perform_eval:
                    logger.info(f"--- Running evaluation at step {global_step} ---")
                    # Pass the tokenizer to the evaluate function
                    metrics = evaluate(tokenizer) # Call evaluate with tokenizer

                    # Log evaluation metrics only on the main process where they are calculated
                    if accelerator.is_main_process and metrics: # Check if metrics dict is not empty
                        eval_log = {f"eval_{k}": v for k, v in metrics.items()}
                        accelerator.log(eval_log, step=global_step)

                        # --- Early Stopping Check (already guarded by is_main_process) ---
                        if early_stopping_enabled:
                            current_metric_value = metrics.get(metric_for_best_model)
                            if current_metric_value is None:
                                logger.warning(f"Metric '{metric_for_best_model}' not found in evaluation results. Skipping early stopping check.") # Fixed indentation
                            elif is_better(current_metric_value, best_metric_value):
                                best_metric_value = current_metric_value # Fixed indentation
                                perform_save = True # Ensure we save if it's the best step model
                                best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}") # Store the actual path for step save
                                logger.info(f"*** New best model found at step {global_step} with {metric_for_best_model}: {best_metric_value:.4f} ***") # Fixed indentation
                            else:
                                logger.info(f"Metric {metric_for_best_model} did not improve at step {global_step}: {current_metric_value:.4f} (Best: {best_metric_value:.4f})") # Fixed indentation


                # --- Saving Trigger (already guarded by is_main_process) ---
                if perform_save:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # Also save the unwrapped PEFT model config and adapters
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                        # --- Checkpoint Limiting ---
                        if save_total_limit and save_total_limit > 0:
                            saved_checkpoints.append(save_path)
                            if len(saved_checkpoints) > save_total_limit:
                                checkpoint_to_delete = saved_checkpoints.pop(0)
                                try:
                                    logger.info(f"Deleting old checkpoint: {checkpoint_to_delete}")
                                    # Use shutil.rmtree for directories
                                    import shutil
                                    shutil.rmtree(checkpoint_to_delete)
                                except OSError as e:
                                    logger.error(f"Error deleting checkpoint {checkpoint_to_delete}: {e}")
                        # --- End Checkpoint Limiting ---


            if global_step >= max_train_steps: # Check using global_step
                break
        # --- End Epoch Step Loop ---

        # --- Epoch End Evaluation/Saving ---
        perform_epoch_eval = False
        perform_epoch_save = False

        # Use the evaluation_strategy variable defined outside the loop
        if evaluation_strategy == "epoch":
            perform_epoch_eval = True
        # Use the save_strategy variable defined outside the loop
        if save_strategy == "epoch":
            perform_epoch_save = True

        if perform_epoch_eval:
            logger.info(f"--- Running evaluation at end of epoch {epoch} (Step {global_step}) ---")
            # Pass the tokenizer to the evaluate function
            metrics = evaluate(tokenizer) # Call evaluate with tokenizer

            # Log and check early stopping only on the main process
            if accelerator.is_main_process and metrics:
                eval_log = {f"eval_{k}": v for k, v in metrics.items()}
                accelerator.log(eval_log, step=global_step) # Log at the global step corresponding to epoch end

                if early_stopping_enabled:
                    current_metric_value = metrics.get(metric_for_best_model)
                    if current_metric_value is None:
                         logger.warning(f"Metric '{metric_for_best_model}' not found in evaluation results. Skipping early stopping check.")
                    elif is_better(current_metric_value, best_metric_value):
                        best_metric_value = current_metric_value
                        perform_epoch_save = True # Ensure we save if it's the best epoch model
                        best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}") # Store the actual path for epoch save
                        logger.info(f"*** New best model found at end of epoch {epoch} with {metric_for_best_model}: {best_metric_value:.4f} ***")
                    else:
                         logger.info(f"Metric {metric_for_best_model} did not improve at epoch {epoch}: {current_metric_value:.4f} (Best: {best_metric_value:.4f})")


        if perform_epoch_save:
             if accelerator.is_main_process:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}") # Use global_step for epoch checkpoint name
                # Only save if it's the best model found so far during this epoch's evaluation, or if early stopping is off
                should_save_epoch_checkpoint = not early_stopping_enabled or (early_stopping_enabled and save_path == best_checkpoint_path)

                if should_save_epoch_checkpoint:
                    accelerator.save_state(save_path)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved epoch checkpoint to {save_path}")

                    # Checkpoint Limiting for epoch saves
                    if save_total_limit and save_total_limit > 0:
                        saved_checkpoints.append(save_path)
                        if len(saved_checkpoints) > save_total_limit:
                            checkpoint_to_delete = saved_checkpoints.pop(0)
                            # Avoid deleting the best checkpoint if it happens to be the oldest
                            if checkpoint_to_delete != best_checkpoint_path:
                                try:
                                    logger.info(f"Deleting old checkpoint: {checkpoint_to_delete}")
                                    import shutil
                                    shutil.rmtree(checkpoint_to_delete)
                                except OSError as e:
                                    logger.error(f"Error deleting checkpoint {checkpoint_to_delete}: {e}")
                            else:
                                logger.info(f"Keeping best checkpoint: {checkpoint_to_delete}")
                                saved_checkpoints.insert(0, checkpoint_to_delete) # Put it back if it was the best

                elif early_stopping_enabled:
                     logger.info(f"Skipping epoch {epoch} checkpoint save as it wasn't the best ({metric_for_best_model}).")


        if global_step >= max_train_steps:
            logger.info("Maximum training steps reached. Exiting training loop.")
            break
    # --- End Training Loop ---


    # --- Final Actions ---
    accelerator.wait_for_everyone() # Ensure all processes finish

    # Load best model if enabled
    if early_stopping_enabled and best_checkpoint_path:
        # Check if best_checkpoint_path is just a step identifier or a full path
        if not os.path.isdir(best_checkpoint_path):
             # Attempt to construct path if only step identifier was stored (e.g., from step-based eval but epoch-based save)
             potential_path = os.path.join(output_dir, f"checkpoint-{best_checkpoint_path.split('_')[-1]}")
             if os.path.isdir(potential_path):
                  best_checkpoint_path = potential_path
             else:
                  logger.warning(f"Could not find best checkpoint directory for identifier: {best_checkpoint_path}. Final model might not be the best.")
                  best_checkpoint_path = None # Reset if path not found

        if best_checkpoint_path and os.path.isdir(best_checkpoint_path):
            logger.info(f"Loading best model from checkpoint: {best_checkpoint_path}")
            accelerator.load_state(best_checkpoint_path)
        else:
             logger.warning("Early stopping was enabled, but no best checkpoint path was found or valid. Saving the final state.")


    # Save final model (which might be the reloaded best model)
    if accelerator.is_main_process:
        final_save_path = os.path.join(config['model']['output_dir'], "final_model")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Saved final model to {final_save_path}")
    # --- End Final Actions ---

    accelerator.end_training() # Cleans up WandB


if __name__ == "__main__":
    main()
