from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel, pipeline

from rlprompt.rewards import BaseReward


SUPPORTED_LMS = ["gpt2"]
SUPPORTED_LEFT_TO_RIGHT_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']
SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


def compute_em_score(prediction: str,
                     truth: str):
    """
    Computes the exact match (EM) score for a list of predicted strings and ground truth strings.

    Args:
        predictions: List of predicted strings.
        truths: List of ground truth strings.

    Returns:
        List of booleans indicating whether each prediction exactly matches its corresponding ground truth.
    """
    # print("\npredictions")
    # print(prediction)
    #
    # print("\ntruths")
    # print(truth)

    return prediction == truth

    # em_score = [p == t for p, t in zip(predictions, truths)]
    # print("\nem_score")
    # print(em_score)
    # return sum(em_score)/len(em_score)


def compute_f1_score(prediction: str,
                     truth: str) -> float:
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

    # precision is ratio of shared words to the total # of words in prediction
    if len(prediction.split()) > 0:
        precision = shared_words_count(prediction, truth) / len(prediction.split())
    else:
        precision = 0

    # recall is ratio of shared words to the total # of words in ground truth
    if len(truth.split()) > 0:
        recall = shared_words_count(prediction, truth) / len(truth.split())
    else:
        recall = 0

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)


class QuestionAnsweringReward(BaseReward):
    """
    Reward class for Question-Answering task.
    """

    def __init__(self, 
                 task_lm: str,
                 compute_zscore: bool):

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        
        assert task_lm in SUPPORTED_LMS
        print('Task LM:', task_lm)
        self.task_lm = task_lm

        self.compute_zscore = compute_zscore
        self.tokenizer = AutoTokenizer.from_pretrained(task_lm)
        self.generator = pipeline("question-answering",
                                  model=self.task_lm,
                                  tokenizer=self.tokenizer,
                                  device=0)

    def forward(self, 
                source_texts: Dict,
                target_labels: List[str],
                output_tokens: List[List[str]],
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

        # convert tokens to output prompts
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        batch_size = len(source_texts)
        # print("\nsource_texts")
        # print(source_texts)

        rewards = []
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for i, (prompt, question, context, label) in enumerate(zip(prompt_strings,
                                                     source_texts['question'],
                                                     source_texts['context'],
                                                     target_labels)):

            # print(prompt)
            full_input = prompt + " " + question
            hypos = self.generator(question=full_input, context=context)
            pred_answer = hypos['answer']

            # compute EM score for each example
            em_scores = compute_em_score(pred_answer, label)
            quantities_to_log['em_scores'].append(torch.as_tensor(em_scores))

            # compute f1 score for each example
            f1_scores = compute_f1_score(pred_answer, label)
            quantities_to_log['f1_scores'].append(torch.as_tensor(f1_scores))

            # combine EM and f1 scores
            reward = em_scores + f1_scores

            rewards.append(reward)

        rewards = torch.as_tensor(rewards)
        # normalize rewards
        if mode == 'train' and self.compute_zscore:
            # print("\nrewards normalizing")
            # print(rewards)
            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()

            rewards = (rewards - mean) / (std + 1e-8)

        # compute overall reward
        # overall_reward = sum(rewards) / len(rewards)

        return rewards, quantities_to_log

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s)
                for s in tokens]
