import argparse
import os
import torch
import yaml
import wandb

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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics
    acc = calculate_accuracy(preds, labels)
    f1 = calculate_f1(preds, labels)
    em = calculate_exact_match(preds, labels)
    mrr = calculate_mrr(preds, labels)
    perplexity = calculate_perplexity(pred.predictions, labels)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'exact_match': em,
        'mrr': mrr,
        'perplexity': perplexity
    }


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
