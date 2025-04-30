\
# -*- coding: utf-8 -*-
""" QA metric calculation functions adapted from TriviaQA evaluation script """
from collections import Counter
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\\b(a|an|the)\\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
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
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculates the maximum score of a metric over all ground truths."""
    scores_for_ground_truths = []
    # Ensure ground_truths is iterable, even if it's a single string
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        # Handle cases where ground_truths might be None or other non-iterable types
        # Or raise an error if this shouldn't happen
        return 0 # Or handle appropriately

    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    # Handle empty ground_truths list to avoid error on max()
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0

# You might want to add a wrapper function here later, e.g.:
# def calculate_qa_metrics(predictions, references):
#     """Calculates average EM and F1 over lists of predictions and references."""
#     total_em = 0
#     total_f1 = 0
#     count = len(predictions)
#     if count == 0:
#         return {'exact_match': 0, 'f1': 0}
#
#     for pred, ref_list in zip(predictions, references):
#         # Assuming ref_list contains the possible ground truths for the prediction
#         total_em += metric_max_over_ground_truths(exact_match_score, pred, ref_list)
#         total_f1 += metric_max_over_ground_truths(f1_score, pred, ref_list)
#
#     return {
#         'exact_match': total_em / count,
#         'f1': total_f1 / count
#     }

