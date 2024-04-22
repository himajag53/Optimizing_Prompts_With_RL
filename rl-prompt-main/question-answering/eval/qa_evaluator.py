from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, pipeline

from qa_reward import compute_em_score, compute_f1_score, compute_length_penalty
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
        no_prompt_answers = []
        questions = []
        gts = []
        f1_scores = []
        no_prompt_f1s = []
        em_scores = []
        no_prompt_ems = []
        len_pens = []
        no_prompt_lens = []

        for i, batch in enumerate(dataloader):
            question = batch['source_texts'][0]
            questions.append(question)

            gt_answer = batch['target_labels'][0]
            gts.append(gt_answer)

            full_input = " ".join((self.prompt, question))
            hypos_prompt = self.generator(text_inputs=full_input,
                                   max_new_tokens=20,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   # Only return generated text, without the prompt
                                   return_full_text=False
                                   )

            hypos_nop = self.generator(text_inputs=question,
                                   max_new_tokens=20,
                                   pad_token_id=self.tokenizer.eos_token_id,
                                   # Only return generated text, without the prompt
                                   return_full_text=False
                                   )

            pred_answer = hypos_prompt[0]
            answers.append(pred_answer['generated_text'])

            pred_answer_np = hypos_nop[0]
            no_prompt_answers.append(pred_answer_np['generated_text'])

            # compute EM score for each example
            em_score = compute_em_score(pred_answer, gt_answer)
            em_scores.append(em_score)
            # compute EM score for no prompt example
            no_prompt_em = compute_em_score(pred_answer_np, gt_answer)
            no_prompt_ems.append(no_prompt_em)

            # compute f1 score for each example
            f1_score = compute_f1_score(pred_answer, gt_answer)
            f1_scores.append(f1_score)
            # compute f1 score for no prompt example
            no_prompt_f1 = compute_f1_score(pred_answer_np, gt_answer)
            no_prompt_f1s.append(no_prompt_f1)

            # compute len penalty
            len_pen = compute_length_penalty(pred_answer, gt_answer)
            len_pens.append(len_pen)
            # compute len penalty for no prompt answer
            no_prompt_len_pen = compute_length_penalty(pred_answer_np, gt_answer)
            no_prompt_lens.append(no_prompt_len_pen)

            if i%200 == 0:
                print()
                print(question)
                print(gt_answer)
                print("prompted")
                print(pred_answer['generated_text'])
                print(f1_score)
                print(em_score)
                print(len_pen)
                print("no prompt")
                print(pred_answer_np['generated_text'])
                print(no_prompt_f1)
                print(no_prompt_em)
                print(no_prompt_len_pen)

            if i >= 3000:
                print("breaking early")
                break

        return (questions, answers, gts, em_scores, f1_scores, len_pens,
                no_prompt_answers, no_prompt_ems, no_prompt_f1s, no_prompt_lens)


