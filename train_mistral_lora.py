import os
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_model(config):
    """Prepare the model with quantization and LoRA configuration"""
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_use_double_quant=config['quantization']['use_double_quant'],
        bnb_4bit_quant_type=config['quantization']['quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization']['compute_dtype']),
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Create LoRA configuration with layer-wise settings
    lora_config = LoraConfig(
        r=config['lora']['basic']['r'],
        lora_alpha=config['lora']['basic']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['basic']['dropout'],
        bias=config['lora']['advanced']['bias'],
        task_type=config['lora']['advanced']['task_type'],
        fan_in_fan_out=config['lora']['advanced']['fan_in_fan_out'],
        modules_to_save=config['lora']['advanced']['modules_to_save'],
        init_lora_weights=config['lora']['advanced']['init_lora_weights'],
        rank_pattern=config['lora']['rank_pattern'],
        alpha_pattern=config['lora']['alpha_pattern'],
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model

def prepare_tokenizer(config):
    """Prepare the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def format_truthfulqa(example):
    """Format TruthfulQA examples for training"""
    question = example['question']
    correct_answer = example['correct_answers'][0] if example['correct_answers'] else ""
    
    # Format: question followed by answer
    formatted_text = f"Question: {question}\nAnswer: {correct_answer}"
    return {"text": formatted_text}

def prepare_dataset(tokenizer, config):
    """Prepare and tokenize the dataset"""
    # Load TruthfulQA dataset
    dataset = load_dataset(config['model']['dataset_name'], "multiple_choice")
    
    # Format the dataset
    formatted_dataset = dataset['validation'].map(format_truthfulqa)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['training']['max_length'],
            padding="max_length",
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        remove_columns=formatted_dataset.column_names,
        batched=True,
    )
    
    return tokenized_dataset

def main():
    # Load configuration
    config = load_config()
    
    # Initialize wandb
    wandb.init(project=config['model']['wandb_project'])
    
    # Prepare model and tokenizer
    model = prepare_model(config)
    tokenizer = prepare_tokenizer(config)
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer, config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training']['num_epochs'],
        warmup_ratio=config['training']['warmup_ratio'],
        logging_steps=config['trainer']['logging_steps'],
        save_strategy=config['trainer']['save_strategy'],
        evaluation_strategy=config['trainer']['evaluation_strategy'],
        report_to="wandb",
        fp16=config['trainer']['fp16'],
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
if __name__ == "__main__":
    main() 