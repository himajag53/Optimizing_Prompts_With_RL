from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from rlprompt.rewards import BaseReward


SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']

SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


def compute_em_score(prediction: dict,
                     truth: str):
    """
    Computes the exact match (EM) score for a list of predicted strings and ground truth strings.

    Args:
        predictions: List of predicted strings.
        truths: List of ground truth strings.

    Returns:
        List of booleans indicating whether each prediction exactly matches its corresponding ground truth.
    """

    return prediction['generated_text'] == truth


def compute_length_penalty(prediction: dict,
                    truth: str):
    """
    Computes difference in length between true answer and prediction
    :param prediction:
    :param truth:
    :return:
    """

    len_t = len(truth)
    len_p = len(prediction['generated_text'])

    if len_p < len_t:
        return 0

    return 1-(len(truth)/len(prediction['generated_text']))


def compute_f1_score(prediction: dict,
                     truth: str):
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

    prediction = prediction['generated_text']
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
                 task_top_k: int,
                 num_samples: int,
                 num_bootstraps: int,
                 compute_zscore: bool,
                 template: str,
                 pad_token: str,
                 lower_outputs: bool,
                 control_output_length: bool = False,
                 end_punct: str = '"'):

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        
        assert task_lm in SUPPORTED_LMS or task_lm in SUPPORTED_MASK_LMS
        print('Task LM:', task_lm)
        self.task_lm = task_lm

        self.top_k = task_top_k
        self.top_p = 1.0
        self.end_punct = end_punct
        self.control_output_length = control_output_length
        self.lower_outputs = lower_outputs
        self.num_samples = num_samples
        self.num_bootstraps = num_bootstraps

        self.compute_zscore = compute_zscore
        self.tokenizer = AutoTokenizer.from_pretrained(task_lm, pad_token=pad_token)
        self.template = template

        self.generator = pipeline("text-generation",
                                  model=self.task_lm,
                                  tokenizer=self.tokenizer,
                                  device=0)

    def compute_sample_rewards(self,
                               source_text: str,
                               generated_texts: List[str],
                               target_label: str
                               ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Compute
        :param source_text:
        :param generated_texts:
        :param target_label:
        :return:
        """

        hypos = generated_texts

        em_scores = []
        f1_scores = []
        len_scores = []

        for pred in hypos:
            em_scores.append(compute_em_score(pred, target_label))
            f1_scores.append(compute_f1_score(pred, target_label))
            len_scores.append(compute_length_penalty(pred, target_label))

        sum_rewards = [(em + f1 + le) / 3
                       for em, f1, le in zip(em_scores, f1_scores, len_scores)]

        return sum_rewards, em_scores, f1_scores, len_scores

    def forward(self, 
                source_texts: List[str],
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

        n_reward = self.num_samples
        k_reward = self.num_bootstraps
        N = n_reward * k_reward

        rewards = []
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        input_rewards: Dict[str, List[float]] = defaultdict(list)

        # loop through the prompts
        for i, (prompt, src, label) in enumerate(zip(prompt_strings,
                                                     source_texts,
                                                     target_labels)):

            full_input = " ".join((prompt, src))
            hypos = self.generator(text_inputs=full_input,
                                   max_new_tokens=10,
                                   top_k=self.top_k,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   num_return_sequences=N,
                                   # Only return generated text, without the prompt
                                   return_full_text=False
                                   )

            sum_rewards, em_scores, f1_scores, len_scores = self.compute_sample_rewards(full_input, hypos, label)
            # pred_answer = hypos[0]

            # compute EM score for each example
            # em_scores = compute_em_score(pred_answer, label)
            quantities_to_log['em_scores'].append(em_scores)

            # compute f1 score for each example
            # f1_scores = compute_f1_score(pred_answer, label)
            quantities_to_log['f1_scores'].append(f1_scores)

            # len_scores = compute_length_penalty(pred_answer, label)

            quantities_to_log['brevity_score'].append(len_scores)

            # combine EM and f1 scores

            # Bootstrap the max reward for k times and average
            bootstrap_max_rewards: List[float] = \
                self._boostrap_max_rewards_k_times(sum_rewards, k_reward)

            # Average boostrap max rewards as the final reward
            reward = torch.Tensor(bootstrap_max_rewards).float().mean()

            # Keep track of each input's max rewards to compute z-score
            input_rewards[src] += bootstrap_max_rewards
            # reward = torch.as_tensor(em_scores) + torch.as_tensor(f1_scores) + torch.as_tensor(len_scores)
            # print(reward)

            rewards.append(reward)

        # print()
        # print(src)
        # print(prompt)
        # print(hypos[0]['generated_text'])
        # print(label)
        # print()

        # print(generated_texts)
        rewards_tensor = torch.stack(rewards)
        # normalize rewards
        if mode == 'train' and self.compute_zscore:

            rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
            rewards_tensor = self._compute_reward_zscores(rewards_tensor,
                                                          source_texts,
                                                          input_rewards)
            # mean = rewards_tensor.mean()
            # std = rewards_tensor.std()
            #
            # rewards = (rewards - mean) / (std + 1e-8)

        # compute overall reward
        # overall_reward = sum(rewards) / len(rewards)

        return rewards_tensor, quantities_to_log

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s)
                for s in tokens]

    def _compute_reward_zscores(
        self,
        rewards_tensor: torch.Tensor,
        input_texts: List[str],
        input_rewards: Dict[str, List[float]],
        eps: float = 1e-4
    ) -> torch.Tensor:
        input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
        input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
        idx_means = torch.tensor([input_reward_means[s] for s in input_texts])
        idx_stds = torch.tensor([input_reward_stds[s] for s in input_texts])

        return (rewards_tensor - idx_means.float()) / (idx_stds.float() + eps)


    def _boostrap_max_rewards_k_times(
        self,
        rewards: List[float],
        k: int
    ) -> List[float]:

        # Segment list rewards into k equal sub-lists
        l = len(rewards)
        assert l % k == 0, f'l={l}, k={k}'
        segmented_rewards = [rewards[i*l//k:(i+1)*l//k]
                             for i in range(k)]  # [k, l/k]
        # We use different rewards for each bootstrap for now
        bootstrap_rewards = segmented_rewards

        # For each sub-list, take the max as the sub-reward
        values, indices = (torch.tensor(bootstrap_rewards)
                           .float().max(dim=1))

        # Take numbers from the original list to avoid numerical issues
        bootstrap_max_rewards = [bootstrap_rewards[i][index]
                                 for i, index in enumerate(indices)]

        return bootstrap_max_rewards

