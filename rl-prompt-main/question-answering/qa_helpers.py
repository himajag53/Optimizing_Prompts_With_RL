from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Tuple

from qa_data_utils import QuestionAnsweringDataset, load_question_answering_dataset
from qa_reward import QuestionAnsweringReward


@dataclass
class QuestionAnsweringRewardConfig():
    task_lm: str = "gpt2"
    compute_zscore: bool = True


@dataclass
class QuestionAnsweringDatasetConfig:
    dataset: str = "squad"
    base_path: str = './data'


def make_question_answering_datasets(config: "DictConfig") -> (
        Tuple)[QuestionAnsweringDataset, QuestionAnsweringDataset, QuestionAnsweringDataset]:
    """
    TODO
    """
    data_dict = {}
    for split in ['train', 'dev', 'test']:

        source_contexts, source_questions, target_labels, context_idx_map = load_question_answering_dataset(
            config.dataset, split, config.base_path)
        tst_dataset = QuestionAnsweringDataset(source_contexts, source_questions, target_labels, context_idx_map)
        data_dict[split] = tst_dataset

    return data_dict['train'], data_dict['dev'], data_dict['test']


def make_question_answering_reward(config: "DictConfig") -> QuestionAnsweringReward:
    """
    TODO
    """
    return QuestionAnsweringReward(config.task_lm, config.compute_zscore)