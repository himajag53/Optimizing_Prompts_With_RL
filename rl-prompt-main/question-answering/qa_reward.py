from typing import List
from rlprompt.rewards import BaseReward


def shared_words_count(string1: str, 
                       string2: str) -> int:
    """
    Counts the number of shared words between two strings.

    Args:
        string1: The first input string.
        string2: The second input string.

    Returns:
        The number of words that are present in both input strings.
    """
    words1 = set(string1.split())
    words2 = set(string2.split())
    return len(words1.intersection(words2))

def compute_em_score(predictions: List[str], 
                     truths: List[str]):
    """
    Computes the exact match (EM) score for a list of predicted strings and ground truth strings.

    Args:
        predictions: A list of predicted strings.
        truths: A list of ground truth strings.

    Returns:
        A list of booleans indicating whether each prediction exactly matches its corresponding ground truth.
    """
    return [p == t for p, t in zip(predictions, truths)]

def compute_f1_score(predictions: List[str], 
                     truths: List[str]):
    """
    TODO
    """

    # precision is ratio of shared words to the total # of words in prediction
    # recall is ratio of shared words to the total # of words in ground truth

    precisions = []
    recalls = []

    for p, t in zip(predictions, truths):
        precisions.append(shared_words_count(p, t) / len(p))
        recalls.append(shared_words_count(p, t) / len(t))

    precisions


class QuestionAnsweringReward(BaseReward):
    """
    TODO
    """

    def __init__(self, 
                 task_lm: str,
                 compute_zscore: bool):
        pass

    def forward(self, 
                source_texts, 
                target_labels, 
                output_tokens, 
                to_tensor, 
                mode):
        """
        TODO
        """

        # sum em score, f1 score

        # normalize across batch

        pass