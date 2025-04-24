import argparse
import os
import torch
import yaml
import wandb

import numpy as np

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from code.config_loader import load_config
from code.data_preparation import prepare_tokenizer, prepare_dataset
from code.model_preparation import prepare_model
from code.metrics import calculate_accuracy, calculate_f1, calculate_exact_match, calculate_mrr, calculate_perplexity


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train Mistral LoRA model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    return parser.parse_args()


def compute_metrics(eval_preds):
    logits, labels = eval_preds.predictions, eval_preds.label_ids

    # Get predicted token IDs (shape: batch_size, seq_len)
    preds = np.argmax(logits, axis=-1)

    # --- IMPORTANT: Filter out padding tokens (-100) ---
    # Create a mask for non-padding labels
    mask = labels != -100

    # Apply the mask to flatten predictions and labels into 1D arrays
    # Only consider positions where the label is not -100
    valid_preds = preds[mask]
    valid_labels = labels[mask]

    metrics = {}

    # --- Calculate Accuracy and F1 on valid (non-padded) tokens ---
    if valid_labels.size > 0: # Avoid division by zero if mask is all False
        try:
            acc = calculate_accuracy(valid_preds, valid_labels)
            f1 = calculate_f1(valid_preds, valid_labels)
            metrics['accuracy'] = acc
            metrics['f1'] = f1
        except Exception as e:
            print(f"Warning: Accuracy/F1 calculation failed: {e}")
            # Optionally set to NaN or skip if calculation fails
            metrics['accuracy'] = np.nan
            metrics['f1'] = np.nan
    else:
        print("Warning: No valid labels found after masking padding.")
        metrics['accuracy'] = 0.0 # Or np.nan
        metrics['f1'] = 0.0 # Or np.nan

    # --- Calculate Perplexity ---
    # Ensure calculate_perplexity handles padding correctly OR pass filtered inputs
    # Option A (Simpler, if calculate_perplexity handles internal masking):
    try:
        # NOTE: Verify if calculate_perplexity in metrics.py correctly ignores labels == -100
        perplexity = calculate_perplexity(logits, labels)
        metrics['perplexity'] = perplexity
    except Exception as e:
        print(f"Warning: Perplexity calculation failed: {e}")
        metrics['perplexity'] = np.nan

    # Option B (More Robust: Filter inputs here if needed):
    # try:
    #   if valid_labels.size > 0:
    #       # We need logits corresponding to valid labels
    #       # This requires careful indexing based on the mask
    #       # Example (might need adjustment based on exact shapes/logic):
    #       # valid_logits = logits[mask] # This might not work directly if logits are 3D
    #       # Need to gather logits corresponding to valid_labels positions
    #       # Placeholder - requires careful implementation if needed:
    #       # gathered_logits = gather_valid_logits(logits, mask)
    #       # perplexity = calculate_perplexity(gathered_logits, valid_labels)
    #
    #       # For now, stick with Option A and assume internal handling or check metrics.py
    #       perplexity = calculate_perplexity(logits, labels) # Revert to Option A logic
    #       metrics['perplexity'] = perplexity
    #   else:
    #      metrics['perplexity'] = np.nan
    # except Exception as e:
    #   print(f"Warning: Perplexity calculation failed: {e}")
    #   metrics['perplexity'] = np.nan


    # --- Remove EM and MRR for now ---
    # These metrics, as implemented in metrics.py, are likely not meaningful
    # when applied directly to token IDs here. Meaningful EM requires decoding.
    # em = calculate_exact_match(valid_preds, valid_labels) # Probably incorrect interpretation
    # mrr = calculate_mrr(valid_preds, valid_labels) # Probably incorrect interpretation
    # metrics['exact_match'] = em
    # metrics['mrr'] = mrr

    return metrics


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load config
    config = load_config(args.config)

    # Init wandb
    wandb.init(project=config['model']['wandb_project'])

    # Prepare model and tokenizer
    model = prepare_model(config)
    tokenizer = prepare_tokenizer(config)

    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer, config)

    # Split the dataset into train and eval
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['validation']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training']['num_epochs'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['trainer']['logging_steps'],
        save_strategy=config['trainer']['save_strategy'],
        eval_strategy=config['trainer']['evaluation_strategy'],
        report_to="wandb",
        fp16=config['trainer']['fp16'],
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize Trainer
    #model = torch.compile(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save final model
    trainer.save_model()


if __name__ == "__main__":
    main()
