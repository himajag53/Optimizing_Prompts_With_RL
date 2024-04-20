import os
import json
from typing import Optional, List, Tuple

from torch.utils.data import Dataset


class QuestionAnsweringDataset(Dataset):
    """
    TODO
    """

    # def __init__(self,
    #              source_contexts,
    #              source_questions,
    #              target_answers,
    #              context_idx_map):
    def __init__(self,
             source_texts,
             target_answers):
        
        assert len(source_texts) == len(target_answers)

        # self.source_contexts = source_contexts
        # self.source_questions = source_questions
        # self.target_answers = target_answers
        # self.context_idx_map = context_idx_map
        self.source_texts = source_texts
        self.target_answers = target_answers

    def __len__(self):
        """
        Get number of questions in the dataset.
        """
        return len(self.source_texts)

    def __getitem__(self, index):
        """
        Get the item at the provided index. Returns a dict containing source_context, source_questions, target_labels
        """

        source_text = self.source_texts[index]
        target_answer = self.target_answers[index]

        item = {'source_texts': source_text,
                'target_labels': target_answer}
        # source_context = self.source_contexts[self.context_idx_map[index]]
        # source_question = self.source_questions[index]
        # item = {'source_texts': source_context + " " + source_question,
        #         'target_labels': self.target_answers[index]}

        # item = {'source_context': self.source_contexts[self.context_idx_map[index]],
        #         'source_questions': self.source_questions[index],
        #         'target_answers': self.target_answers[index]}

        return item
    

def load_question_answering_dataset(
    dataset: str,
    split: str,
    base_path: str,
    version: int=4
# ) -> Tuple[List[str], List[str], List[str], dict]:
) -> Tuple[List[str], List[str]]:
    """
    Load in the dataset file.

    :param dataset: dataset source (e.g. 'squad')
    :param split: train or dev
    :param base_path: base path to data directory
    :return: Tuple containing: list of question contexts,
                               list of questions,
                               list of answers,
                               dict mapping each question to its context
    """

    # make all the sources, target and context idx dict
    assert dataset in ['squad']

    # get file path info
    filepath = f'{dataset}/{split}-v{version}.json'
    full_filepath = os.path.join(base_path, filepath)

    # read file
    with open(full_filepath, 'r') as f:
        data_json = json.load(f)

    # set up lists for storing
    source_contexts = []
    source_questions = []
    context_idx_map = {}

    source_texts = []
    target_labels = []


    question_idx = 0
    for question in data_json:
        if version == 3:
            source_texts.append(question['context_and_question'])
        else:
            source_texts.append(question['question'])
            source_contexts.append(question['context'])
        target_labels.append(question['answer'])

    # for i, context in enumerate(data_json):
    #     # source_contexts.append(context['context'])
    #
    #     for qa in context['qas']:
    #         source_questions.append(qa['question'])
    #         target_labels.append(qa['answers'][0]['text'])
    #         context_idx_map[question_idx] = i
    #         qa['context'] = context['context']
    #         source_texts.append(qa)
    #         question_idx += 1

    # return source_contexts, source_questions, target_labels, context_idx_map
    return source_texts, target_labels


def preprocess_input():
    """
    TODO
    """
    pass


