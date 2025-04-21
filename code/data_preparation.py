from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def prepare_tokenizer(config):
    """Prepare the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_triviaqa(example):
    """Format TriviaQA examples for training"""
    question = example['question']

    # TriviaQA has answers as a list, we'll use the first answer
    answer = example['answer']['value'] if isinstance(example['answer'], dict) else example['answer']['aliases'][0]

    # Create instruction format
    return {"text": f"Question: {question}\nAnswer: {answer}"}


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

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config['training']['max_length'],
            padding="max_length",
        )

    # Tokenize both sets
    print("Tokenizing datasets...")
    tokenized_train = formatted_train.map(
        tokenize_function,
        remove_columns=formatted_train.column_names,
        batched=True,
    )

    tokenized_validation = formatted_val.map(
        tokenize_function,
        remove_columns=formatted_val.column_names,
        batched=True,
    )

    print(f"Final training examples: {len(tokenized_train)}")
    print(f"Final validation examples: {len(tokenized_validation)}")

    # Return as DatasetDict
    return DatasetDict({
        'train': tokenized_train,
        'validation': tokenized_validation
    })