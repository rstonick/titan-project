import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def calculate_accuracy(predictions, labels):
    return accuracy_score(labels, predictions)


def calculate_f1(predictions, labels):
    return f1_score(labels, predictions, average='weighted')


def calculate_exact_match(predictions, labels):
    return np.mean([1 if p == l else 0 for p, l in zip(predictions, labels)])


def calculate_mrr(predictions, labels):
    mrr_total = 0
    for pred, label in zip(predictions, labels):
        if label in pred:
            rank = pred.index(label) + 1
            mrr_total += 1 / rank
    return mrr_total / len(labels)


def calculate_perplexity(logits, labels):
    # Assuming logits are the raw output from the model before softmax
    # and labels are the true indices
    log_probs = -np.log(np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))
    nll = np.take_along_axis(log_probs, np.expand_dims(labels, axis=-1), axis=-1).squeeze()
    perplexity = np.exp(np.mean(nll))
    return perplexity 