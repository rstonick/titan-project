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
from code.metrics import calculate_qa_metrics, calculate_perplexity # Use the updated functions

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from code.config_loader import load_config
from code.data_preparation import prepare_tokenizer, prepare_dataset
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

    # --- Prepare Dataset and Dataloaders ---
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer, config)

    # Split the dataset into train and eval
    if 'validation' not in tokenized_dataset:
         # If no validation split exists, create one (e.g., 10% of train)
         logger.warning("No validation split found. Creating one from 10% of the training data.")
         split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=config.get('seed', 42))
         train_dataset = split_dataset['train']
         eval_dataset = split_dataset['test']
         
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']


    # --- DataLoader Setup ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Read num_workers from config, default to 0 if not specified
    num_workers = config['trainer'].get('dataloader_num_workers', 0)
    logger.info(f"Using {num_workers} dataloader workers.") # Log the value being used

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config['training']['batch_size'], # Per device batch size
        num_workers=num_workers # <-- Added this line
    )

    # Eval dataloader (Uncommented and configured)
    eval_batch_size = config['trainer'].get('per_device_eval_batch_size', config['training']['batch_size'])
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=eval_batch_size, # Use configured eval batch size
        num_workers=num_workers # <-- Added this line
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
        all_labels_decoded = []

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            gathered_loss = accelerator.gather_for_metrics(loss.reshape(1, -1))
            losses.append(gathered_loss)

            # --- Add prediction generation and gathering ---
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            # Gather preds and labels across devices
            gathered_preds = accelerator.gather_for_metrics(preds)
            gathered_labels = accelerator.gather_for_metrics(labels)

            # Decode tokens (handle padding -100) on the main process after gathering
            if accelerator.is_main_process:
                # Ensure tensors are on CPU and are numpy arrays for decoding
                gathered_preds_np = gathered_preds.cpu().numpy()
                gathered_labels_np = gathered_labels.cpu().numpy()

                # Replace -100 with pad_token_id before decoding
                gathered_labels_np[gathered_labels_np == -100] = eval_tokenizer.pad_token_id
                # Also replace padding in predictions if necessary (though argmax shouldn't yield -100)
                gathered_preds_np[gathered_preds_np == -100] = eval_tokenizer.pad_token_id

                decoded_preds_batch = eval_tokenizer.batch_decode(gathered_preds_np, skip_special_tokens=True)
                decoded_labels_batch = eval_tokenizer.batch_decode(gathered_labels_np, skip_special_tokens=True)

                # Clean up whitespace and store
                all_preds_decoded.extend([p.strip() for p in decoded_preds_batch])
                all_labels_decoded.extend([l.strip() for l in decoded_labels_batch])
            # --- End prediction generation and gathering ---

        # Calculate Loss and Perplexity (on main process after loop)
        metrics = {}
        if accelerator.is_main_process:
            losses = torch.cat(losses)
            try:
                # Truncate losses to the size of the dataset if needed
                if len(losses) > len(eval_dataset):
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
            metrics["loss"] = eval_loss.item()
            metrics["perplexity"] = perplexity

            # --- Calculate QA Metrics (EM, F1) ---
            if all_preds_decoded and all_labels_decoded:
                try:
                    # Ensure labels are in the correct format (list of strings or list of lists)
                    # For TriviaQA, it's usually a list of lists of possible answers.
                    # The current prepare_dataset likely flattens this.
                    # Assuming single answer string for now based on previous code.
                    # If multiple answers, data prep and this call need adjustment.
                    qa_scores = calculate_qa_metrics(all_preds_decoded, all_labels_decoded)
                    metrics.update(qa_scores) # Adds 'exact_match' and 'f1'
                    logger.info(f"Evaluation EM: {metrics.get('exact_match', 'N/A'):.4f}, F1: {metrics.get('f1', 'N/A'):.4f}")
                except Exception as e:
                    logger.error(f"Could not calculate QA metrics (EM/F1): {e}")
                    metrics['exact_match'] = np.nan
                    metrics['f1'] = np.nan
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
                                 logger.warning(f"Metric '{metric_for_best_model}' not found in evaluation results. Skipping early stopping check.")
                            elif is_better(current_metric_value, best_metric_value):
                                best_metric_value = current_metric_value
                                # Save the best model checkpoint if eval strategy is steps
                                if save_strategy == "steps":
                                    perform_save = True # Trigger save because it's the best model so far
                                    logger.info(f"*** New best model found at step {global_step} with {metric_for_best_model}: {best_metric_value:.4f} ***")
                                else: # If save strategy is epoch, just note the best step
                                     best_checkpoint_path = f"step_{global_step}" # Store identifier, actual save happens at epoch end
                                     logger.info(f"*** New best model identified at step {global_step} with {metric_for_best_model}: {best_metric_value:.4f} (will save at epoch end) ***")
                            else:
                                 logger.info(f"Metric {metric_for_best_model} did not improve: {current_metric_value:.4f} (Best: {best_metric_value:.4f})")


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
