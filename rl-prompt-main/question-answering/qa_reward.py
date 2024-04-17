from typing import Dict, List

import numpy as np
import torch
from rlprompt.rewards import BaseReward


SUPPORTED_LMS = []


def compute_em_score(predictions: List[str], 
                     truths: List[str]):
    """
    Computes the exact match (EM) score for a list of predicted strings and ground truth strings.

    Args:
        predictions: List of predicted strings.
        truths: List of ground truth strings.

    Returns:
        List of booleans indicating whether each prediction exactly matches its corresponding ground truth.
    """
    return [p == t for p, t in zip(predictions, truths)]


def compute_f1_score(predictions: List[str], 
                     truths: List[str]) -> float:
    """
    Computes the F1 score for a list of predictions and ground truths.

    Args:
        predictions: List of predicted strings.
        truths: List of ground truth strings.

    Returns:
        f1_score: The F1 score.
    """

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

    # initialize
    precisions = []
    recalls = []

    for p, t in zip(predictions, truths):
        # precision is ratio of shared words to the total # of words in prediction
        if len(p.split()) > 0:
            precisions.append(shared_words_count(p, t) / len(p.split()))
        else:
            precisions.append(0)

        # recall is ratio of shared words to the total # of words in ground truth
        if len(t.split()) > 0:
            recalls.append(shared_words_count(p, t) / len(t.split()))
        else:
            recalls.append(0)

    # compute average precision and recall scores
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)

    # compute f1 score
    if avg_precision + avg_recall == 0:
        return  0
    
    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)


class QuestionAnsweringReward(BaseReward):
    """
    Reward class for Question-Answering task.
    """

    def __init__(self, 
                 task_lm: str,
                 compute_zscore: bool):
        
        assert task_lm in SUPPORTED_LMS
        print('Task LM:', task_lm)
        self.task_lm = task_lm

        self.compute_zscore = compute_zscore

    def forward(self, 
                source_contexts: List[str],
                source_questions: List[str], 
                target_labels: List[str],
                output_tokens: List[str], 
                to_tensor,
                mode):
        """
        Forward pass of the reward function.

        Args:
            source_contexts: List of source contexts.
            source_questions: List of source questions.
            target_labels: List of target labels.
            output_tokens: List of output tokens.
            to_tensor: Function to convert to tensor.
            mode: Training mode ('train' or 'infer').

        Returns:
            overall_reward: Overall reward (average of normalized rewards).
            rewards: List of computed rewards for each input example.
        """

        # compute EM score for each example
        em_scores = [compute_em_score([o], [t]) for o, t in zip(output_tokens, target_labels)]

        # compute f1 score for each example
        f1_scores = [compute_f1_score([o], [t]) for o, t in zip(output_tokens, target_labels)]

        # combine EM and f1 scores
        rewards = [em + f1 for em, f1 in zip(em_scores, f1_scores)]

        # normalize rewards
        if mode == 'train' and self.compute_zscore:
            rewards_tensor = to_tensor(rewards)
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()

            rewards = [(r - mean) / (std + 1e-8) for r in rewards]
        
        # compute overall reward
        overall_reward = sum(rewards) / len(rewards)

        return overall_reward, rewards
    