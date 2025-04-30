import numpy as np # Add numpy import for searching
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def prepare_tokenizer(config):
    """Prepare the tokenizer"""
    # Initialize with padding_side='left' as per warning
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'], 
        padding_side='left', # Set padding side during initialization
        use_fast=True, # Use fast tokenizer for better performance
        #add_eos_token=True #add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_triviaqa(example):
    """Format TriviaQA examples for training"""
    question = example['question']

    # TriviaQA has answers as a list, we'll use the first answer
    answer = example['answer']['value'] if isinstance(example['answer'], dict) else example['answer']['aliases'][0]

    # Create instruction format
    return {"text": f"Question: {question}\nAnswer: {answer}"}


def tokenize_function(examples, tokenizer, max_length):
    # Tokenize the full text (Question + Answer)
    # Ensure EOS token is added if tokenizer doesn't do it by default
    # Some tokenizers add it automatically, others might need explicit instruction
    # or post-processing. For now, assume tokenizer handles it or add it manually if needed.
    full_text = [text + tokenizer.eos_token for text in examples["text"]] # Add EOS token

    model_inputs = tokenizer(
        full_text, # Use text with EOS added
        max_length=max_length,
        padding=False, # Padding will be handled by the DataCollator
        truncation=True
    )

    # For Causal LM, the labels are typically the input_ids themselves.
    # The DataCollatorForLanguageModeling will handle shifting them later.
    # We don't need to manually create or pad labels here.
    # Just return the tokenized inputs.
    # model_inputs["labels"] = model_inputs["input_ids"].copy() # No longer needed here

    # Remove all previous logic for finding separator and masking
    # --- Removed separator search ---
    # --- Removed label masking loop ---
    # --- Removed filtering logic based on processing success ---

    return model_inputs


def prepare_dataset(tokenizer, config, test_size=0.1, max_question_length=100):
    """Prepare and tokenize the dataset with train/validation split"""
    # Load a subset of TriviaQA dataset
    # subset_size = config.get('dataset', {}).get('subset_size', 5000)  # Default to 5000 examples if not specified
    subset_size = config['dataset']['subset_size']

    print(f"Loading {subset_size} examples from TriviaQA dataset...")

    # Load TriviaQA dataset with subset specification
    train_test_split = load_dataset(
        config['dataset']['dataset_name'],
        "rc",
        split=f"train[:{subset_size}]"  # Only take first N examples
    ).train_test_split(test_size=test_size)  # 10% for validation

    print(f"Loaded {len(train_test_split['train'])} training examples and {len(train_test_split['test'])} validation examples")

    # Filter to remove examples with very long questions if needed
    def is_reasonable_length(example):
        return len(example['question']) <= max_question_length

    filtered_train = train_test_split['train'].filter(is_reasonable_length)
    filtered_val = train_test_split['test'].filter(is_reasonable_length)

    print(f"After filtering: {len(filtered_train)} training examples and {len(filtered_val)} validation examples")

    train_columns = filtered_train.column_names
    val_columns = filtered_val.column_names

    # Format the dataset
    formatted_train = filtered_train.map(
        format_triviaqa,
        remove_columns=train_columns
    )

    formatted_val = filtered_val.map(
        format_triviaqa,
        remove_columns=val_columns
    )

    # Get max_length from config
    max_len = config['training']['max_length']

    # Tokenize both sets using the simplified function
    print("Tokenizing datasets (simplified for Causal LM)...")
    from functools import partial
    tokenize_with_args = partial(tokenize_function, tokenizer=tokenizer, max_length=max_len)

    # Ensure load_from_cache_file=False is used if you want to force re-tokenization
    tokenized_train = formatted_train.map(
        tokenize_with_args,
        remove_columns=formatted_train.column_names,
        batched=True,
        load_from_cache_file=False # Keep this to ensure the new function runs
    )

    tokenized_validation = formatted_val.map(
        tokenize_with_args,
        remove_columns=formatted_val.column_names,
        batched=True,
        load_from_cache_file=False # Keep this to ensure the new function runs
    )

    print(f"Final training examples: {len(tokenized_train)}")
    print(f"Final validation examples: {len(tokenized_validation)}")

    # Return as DatasetDict
    return DatasetDict({
        'train': tokenized_train,
        'validation': tokenized_validation
    })