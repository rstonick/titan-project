# /home/hice1/rstonick3/scratch/titan-project/code/prepare_dataset_local.py

import json
import os
from datasets import Dataset, DatasetDict
from accelerate.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

def prepare_local_triviaqa_rc_nocontext(tokenizer, config):
    """
    Loads and prepares the TriviaQA rc.nocontext dataset from local JSON files.

    Args:
        tokenizer: The tokenizer to use.
        config: The configuration dictionary, expected to have keys like:
                config['dataset']['local_path']: Path to the root of the downloaded TriviaQA dataset.
                config['dataset']['train_files']: List of train JSON file paths relative to local_path.
                config['dataset']['validation_files']: List of validation JSON file paths relative to local_path.
                config['dataset']['max_length']: Max sequence length for tokenization.
                config['dataset']['prompt_format']: String format for the input (e.g., "Question: {q}\nAnswer: {a}").

    Returns:
        A tokenized DatasetDict with 'train' and 'validation' splits,
        including 'input_ids', 'attention_mask', 'labels', and 'normalized_aliases'.
    """
    dataset_config = config.get('dataset', {})
    local_path = dataset_config.get('local_path')
    train_files = dataset_config.get('train_files', ['qa/wikipedia-train.json']) # Default example
    validation_files = dataset_config.get('validation_files', ['qa/wikipedia-dev.json']) # Default example
    max_length = dataset_config.get('max_length', 512)
    # Example prompt format, adjust as needed
    prompt_format = dataset_config.get('prompt_format', "Question: {question}\nAnswer: {answer_value}")

    if not local_path:
        raise ValueError("config['dataset']['local_path'] must be set to the TriviaQA directory.")

    base_path = Path(local_path)

    def load_and_extract(relative_files):
        """Loads JSON files and extracts relevant fields."""
        questions = []
        answer_values = []
        normalized_aliases_list = []

        for rel_file in relative_files:
            file_path = base_path / rel_file
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue
            logger.info(f"Loading data from: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # The main data is usually under a 'Data' key
                    items = data.get('Data', [])
                    if not items:
                         logger.warning(f"No 'Data' key found or empty list in {file_path}")
                         # Try loading directly if it's just a list
                         if isinstance(data, list):
                             items = data
                         else:
                             continue # Skip if format is unexpected

                    for item in items:
                        question = item.get('Question')
                        answer_obj = item.get('Answer')

                        if question and answer_obj:
                            answer_value = answer_obj.get('Value')
                            norm_aliases = answer_obj.get('NormalizedAliases')

                            # Ensure we have the necessary fields
                            if answer_value is not None and norm_aliases is not None:
                                questions.append(question)
                                answer_values.append(answer_value)
                                normalized_aliases_list.append(norm_aliases)
                            else:
                                logger.debug(f"Skipping item due to missing Question, Answer.Value, or Answer.NormalizedAliases: {item.get('QuestionId', 'N/A')}")
                        else:
                             logger.debug(f"Skipping item due to missing Question or Answer object: {item.get('QuestionId', 'N/A')}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {file_path}")
            except Exception as e:
                 logger.error(f"Error processing file {file_path}: {e}")

        if not questions:
             logger.warning(f"No valid data extracted from files: {relative_files}")
             # Return empty lists to avoid downstream errors, or raise an error
             return [], [], []


        return questions, answer_values, normalized_aliases_list

    # Load train and validation data
    train_q, train_a, train_aliases = load_and_extract(train_files)
    val_q, val_a, val_aliases = load_and_extract(validation_files)

    # Create Dataset objects
    # Ensure all lists have the same length before creating the Dataset
    if not (len(train_q) == len(train_a) == len(train_aliases)):
        raise RuntimeError(f"Mismatch in lengths for training data: Q={len(train_q)}, A={len(train_a)}, Aliases={len(train_aliases)}")
    if not (len(val_q) == len(val_a) == len(val_aliases)):
         raise RuntimeError(f"Mismatch in lengths for validation data: Q={len(val_q)}, A={len(val_a)}, Aliases={len(val_aliases)}")


    # Handle empty datasets gracefully if needed, though erroring might be better
    if not train_q:
        logger.error("No training data loaded. Cannot proceed.")
        # Depending on requirements, you might want to create an empty dataset
        # or raise an error. Raising error is safer.
        raise ValueError("Failed to load any training data.")
    if not val_q:
         logger.warning("No validation data loaded. Proceeding without validation.")
         # Create empty validation set or handle as needed
         # For simplicity, let's create an empty one matching train schema
         val_dataset = Dataset.from_dict({'question': [], 'answer_value': [], 'normalized_aliases': []})
    else:
         val_dataset = Dataset.from_dict({
             'question': val_q,
             'answer_value': val_a,
             'normalized_aliases': val_aliases
         })

    train_dataset = Dataset.from_dict({
        'question': train_q,
        'answer_value': train_a,
        'normalized_aliases': train_aliases
    })


    # Combine into DatasetDict
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    logger.info(f"Raw dataset loaded: {raw_datasets}")

    # --- Apply subset_size --- NEW SECTION
    subset_size = dataset_config.get('subset_size')
    if subset_size:
        logger.info(f"Applying subset_size: {subset_size}")
        if subset_size > 0:
            # Ensure subset size is not larger than the dataset
            train_subset_size = min(subset_size, len(raw_datasets["train"]))
            val_subset_size = min(subset_size, len(raw_datasets["validation"])) # Apply to validation too for consistency

            logger.info(f"Selecting {train_subset_size} examples for training.")
            raw_datasets["train"] = raw_datasets["train"].select(range(train_subset_size))

            if val_subset_size > 0:
                 logger.info(f"Selecting {val_subset_size} examples for validation.")
                 raw_datasets["validation"] = raw_datasets["validation"].select(range(val_subset_size))
            else:
                 logger.warning("Validation set is empty after subset selection attempt.")
        else:
            logger.warning(f"subset_size ({subset_size}) is not positive. Ignoring.")

    logger.info(f"Dataset after subset selection: {raw_datasets}")
    # --- End subset_size --- 

    # --- Tokenization ---
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Adding EOS token as pad token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Make sure model's config reflects this if saving later
        # config['model']['pad_token_id'] = tokenizer.eos_token_id

    def tokenize_function(examples):
        # Construct the full input string including the chosen answer
        inputs = [
            prompt_format.format(question=q, answer_value=a) + tokenizer.eos_token
            for q, a in zip(examples['question'], examples['answer_value'])
        ]

        # Tokenize the full input
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            padding="max_length", # Pad to max_length
            truncation=True,
            return_tensors=None # Return lists
        )

        # Create labels by copying input_ids
        labels = [list(ids) for ids in model_inputs["input_ids"]] # Make mutable copies

        # Construct the prompt part *without* the answer to find its length
        prompts = [
             prompt_format.format(question=q, answer_value="").split("{answer_value}")[0] # Get text before answer
             for q in examples['question']
        ]
        # Tokenize prompts to find length to mask
        prompt_token_lengths = [
            len(tokenizer(p, add_special_tokens=False).input_ids)
            for p in prompts
        ]

        # Mask prompt tokens in labels
        for i in range(len(labels)):
            prompt_len = prompt_token_lengths[i]
            # Mask tokens up to and including the prompt part
            # Be careful with off-by-one errors depending on tokenizer behavior
            # Usually, mask up to prompt_len
            for j in range(prompt_len):
                 if j < len(labels[i]): # Ensure index is within bounds
                     labels[i][j] = -100

            # Also mask padding tokens in labels
            # Find the first pad token id
            try:
                pad_token_index = model_inputs["input_ids"][i].index(tokenizer.pad_token_id)
                for k in range(pad_token_index, len(labels[i])):
                    labels[i][k] = -100
            except ValueError:
                # No padding token found, sequence might be exactly max_length or shorter
                pass


        model_inputs["labels"] = labels
        # Keep the normalized_aliases column!
        model_inputs["normalized_aliases"] = examples["normalized_aliases"]
        return model_inputs

    # Apply tokenization
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names, # Remove original text columns
        desc="Running tokenizer on dataset",
    )

    logger.info(f"Tokenized dataset created: {tokenized_datasets}")
    logger.info(f"Columns in tokenized dataset: {tokenized_datasets['train'].column_names}")


    return tokenized_datasets
