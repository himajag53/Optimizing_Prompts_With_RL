from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from qa_reward import compute_em_score, compute_f1_score
from rlprompt.rewards import BaseReward


SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                               'gpt2-large', 'gpt2-xl']

SUPPORTED_MASK_LMS = ['distilroberta-base', 'roberta-base', 'roberta-large']


class QuestionAnsweringEvaluator:
    """
    TODO
    """

    def __init__(self,
                 task_lm: str,
                 prompt: str,
                 pad_token: str,
                 lower_outputs: bool,
                 control_output_length: bool = False,
                 end_punct: str = '"'):
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        assert task_lm in SUPPORTED_LMS or task_lm in SUPPORTED_MASK_LMS
        print('Task LM:', task_lm)
        self.task_lm = task_lm
        self.end_punct = end_punct
        self.control_output_length = control_output_length
        self.lower_outputs = lower_outputs

        self.tokenizer = AutoTokenizer.from_pretrained(task_lm, pad_token=pad_token)

        self.generator = pipeline("text-generation",
                                  model=self.task_lm,
                                  tokenizer=self.tokenizer,
                                  device=0)
        self.prompt = prompt

    def evaluate_output(self, dataloader):
        """
        TODO
        """
        answers = []
        questions = []
        gts = []
        f1_scores = []
        em_scores = []

        for i, batch in enumerate(dataloader):
            question = batch['source_texts'][0]
            questions.append(question)

            gt_answer = batch['target_labels'][0]
            gts.append(gt_answer)

            full_input = " ".join((self.prompt, question))
            hypos = self.generator(text_inputs=full_input,
                                   max_new_tokens=50,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   # Only return generated text, without the prompt
                                   return_full_text=False
                                   )

            pred_answer = hypos[0]
            answers.append(pred_answer['generated_text'])

            # compute EM score for each example
            em_score = compute_em_score(pred_answer, gt_answer)
            em_scores.append(em_score)

            # compute f1 score for each example
            f1_score = compute_f1_score(pred_answer, gt_answer)
            f1_scores.append(f1_score)

            if i%200 == 0:
                print()
                print(question)
                print(gt_answer)
                print(pred_answer['generated_text'])
                print(f1_score)
                print(em_score)

            if i >= 2000:
                print("breaking early")
                break

        return questions, answers, gts, em_scores, f1_scores


