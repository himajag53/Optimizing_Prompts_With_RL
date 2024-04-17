import os
import json
from typing import Optional, List, Tuple

from torch.utils.data import Dataset


class QuestionAnsweringDataset(Dataset):
    """
    TODO
    """

    def __init__(self, 
                 source_contexts,
                 source_questions, 
                 target_labels,
                 context_idx_map):
        
        assert len(source_questions) == target_labels

        self.source_contexts = source_contexts
        self.source_questions = source_questions
        self.target_labels = target_labels
        self.context_idx_map = context_idx_map

    def __len__(self):
        """
        TODO
        """
        return len(self.source_questions)

    def __getitem__(self, index):
        """
        TODO
        """
        item = {'source_context': self.source_contexts[self.context_idx_map[index]],
                'source_questions': self.source_questions[index],
                'target_labels': self.target_labels[index]}
        return item
    

def load_question_answering_dataset(
    dataset: str,
    split: str,
    dataset_seed: Optional[int],
    base_path: str
) -> Tuple[List[str], List[str], List[str], dict]:
    """
    TODO
    """
    # make all the sources, target and context idx dict
    assert dataset in ['squad']

    # get file path info
    # seed_dict = {0: '16-100', 1: '16-13', 2: '16-21', 3: '16-42', 4: '16-87'}
    # seed_path = seed_dict[dataset_seed]
    # filepath = f'{dataset}/{seed_path}/{split}.tsv'

    filepath = f'{dataset}'
    full_filepath = os.path.join(base_path, filepath)

    # read file
    with open(full_filepath, 'r') as f:
        data_json = json.load(f)

    # set up lists for storing
    source_contexts = []
    source_questions = []
    target_labels = []
    context_idx_map = {}

    question_idx = 0
    for context, i in enumerate(data_json):
        source_contexts.append(context['context'])

        for qa in context['qas']:
            source_questions.append(qa['question'])
            target_labels.append(qa['answers']['text'])
            context_idx_map[question_idx] = i
            question_idx += 1

    return source_contexts, source_questions, target_labels, context_idx_map


def preprocess_input():
    """
    TODO
    """
    pass


