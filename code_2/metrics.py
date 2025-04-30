import numpy as np
from sklearn.metrics import accuracy_score # Keep if calculate_accuracy is still used elsewhere
import re
import string
from collections import Counter


def calculate_accuracy(predictions, labels):
    # This calculates token-level accuracy if predictions/labels are token IDs
    # Or sequence-level exact match if predictions/labels are full sequences/classes
    # Keep its purpose clear based on where it's called.
    return accuracy_score(labels, predictions)


# --- SQuAD-style F1 and EM Calculation ---

# Helper functions for SQuAD metric normalization (often part of metric implementations)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score_token_overlap(prediction, ground_truth):
    """Calculates F1 score based on token overlap between prediction and ground_truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculates Exact Match score after normalization."""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculates the maximum score of a metric over all ground truths."""
    scores_for_ground_truths = []
    # Ensure ground_truths is always a list, even if only one answer is provided
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        # Handle potential other types or raise an error
        ground_truths = list(ground_truths) # Attempt conversion

    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def calculate_qa_metrics(predictions, references):
    """
    Calculates SQuAD-style F1 and Exact Match.

    Args:
        predictions (list[str]): A list of predicted answer strings.
        references (list[str] or list[list[str]]): A list of ground truth answer strings.
                                                    Can be a list of lists if multiple answers are possible.

    Returns:
        dict: A dictionary containing 'exact_match' and 'f1'.
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match.")

    f1 = exact_match = 0
    for i, (pred, gold_answers) in enumerate(zip(predictions, references)): # Added index i
         # Ensure gold_answers is a list for metric_max_over_ground_truths
        if isinstance(gold_answers, str):
            current_references = [gold_answers]
        else:
            current_references = gold_answers # Assume it's already a list/iterable

        # Calculate scores for this example
        current_em = metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=current_references
        )
        current_f1 = metric_max_over_ground_truths(
            f1_score_token_overlap, prediction=pred, ground_truths=current_references
        )

        # --- Debugging Print ---
        # Print if EM is 0 or F1 is > 1.0 for the first few examples
        # Note: current_f1 should be between 0.0 and 1.0 before scaling
        if (current_em == 0 or current_f1 > 1.0) and i < 10: # Print for first 10 non-matches or potential high F1s
             normalized_pred = normalize_answer(pred)
             normalized_refs = [normalize_answer(ref) for ref in current_references]
             print(f"--- Example {i} (EM={current_em}, F1={current_f1:.4f}) ---") # Show current_f1
             print(f"Raw Pred: '{pred}'")
             print(f"Raw Refs: {current_references}")
             print(f"Norm Pred: '{normalized_pred}'")
             print(f"Norm Refs: {normalized_refs}")
             print("-" * 20)
        # --- End Debugging Print ---

        exact_match += current_em
        f1 += current_f1


    exact_match = 100.0 * exact_match / len(predictions)
    f1 = 100.0 * f1 / len(predictions)

    return {'exact_match': exact_match, 'f1': f1}


def calculate_mrr(predictions, labels):
    # Keep MRR as is, assuming it's used for a ranking task
    # or predictions is a list of ranked items.
    mrr_total = 0
    count = 0
    for pred_list, label in zip(predictions, labels):
        # Ensure pred_list is iterable
        if not hasattr(pred_list, '__iter__') or isinstance(pred_list, str):
             print(f"Warning: Skipping MRR calculation for non-iterable prediction: {pred_list}")
             continue # Skip this item or handle appropriately

        try:
            # Find the rank of the label within the prediction list
            rank = list(pred_list).index(label) + 1
            mrr_total += 1 / rank
            count += 1
        except ValueError:
            # Label not found in the prediction list, rank is effectively infinity, contribution is 0
            pass # Or handle as needed

    return (mrr_total / count) if count > 0 else 0.0


def calculate_perplexity(logits, labels):
    # WARNING: This manual calculation might be unstable and doesn't handle padding (-100) correctly.
    # It's generally better to use the loss returned by the model/Trainer.
    # If you must use this, ensure labels are filtered for padding tokens first.

    # Filter out padding tokens (-100)
    mask = labels != -100
    valid_labels = labels[mask]
    valid_logits = logits[mask] # Ensure logits are filtered consistently

    if valid_labels.size == 0:
        print("Warning: No valid labels found for perplexity calculation after filtering padding.")
        return np.nan # Or some other indicator of invalid calculation

    # Apply softmax to get probabilities
    exp_logits = np.exp(valid_logits - np.max(valid_logits, axis=-1, keepdims=True)) # Improve stability
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Gather the probabilities corresponding to the true labels
    true_label_probs = np.take_along_axis(probs, np.expand_dims(valid_labels, axis=-1), axis=-1).squeeze()

    # Avoid log(0)
    true_label_probs = np.maximum(true_label_probs, 1e-9)

    # Calculate negative log likelihood
    nll = -np.log(true_label_probs)

    # Calculate perplexity
    perplexity = np.exp(np.mean(nll))
    return perplexity