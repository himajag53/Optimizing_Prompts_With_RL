from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Tuple

from qa_data_utils import QuestionAnsweringDataset, load_question_answering_dataset
from qa_reward import QuestionAnsweringReward


@dataclass
class QuestionAnsweringRewardConfig:
    task_lm: str = "distilgpt2"
    task_top_k: int = 5
    num_samples: int = 16
    num_bootstraps: int = 3
    compute_zscore: bool = True
    template: str = '{prompt} "{sentence_1}" "'
    pad_token: str = '<|endoftext|>'
    lower_outputs: bool = False  # Whether to convert all outputs to lower case
    control_output_length: bool = False
    end_punct: str = '"'


@dataclass
class QuestionAnsweringDatasetConfig:
    dataset: str = "squad"
    base_path: str = './data'
    version: int = 3


def make_question_answering_datasets(config: "DictConfig") -> (
        Tuple)[QuestionAnsweringDataset, QuestionAnsweringDataset, QuestionAnsweringDataset]:
    """
    TODO
    """
    data_dict = {}
    for split in ['train', 'dev', 'test']:

        # source_contexts, source_questions, target_labels, context_idx_map = load_question_answering_dataset(
        #     config.dataset, d_split, config.base_path)
        # tst_dataset = QuestionAnsweringDataset(source_contexts, source_questions, target_labels, context_idx_map)

        source_texts, target_labels = load_question_answering_dataset(
            config.dataset, split, config.base_path, config.version)
        tst_dataset = QuestionAnsweringDataset(source_texts, target_labels)
        data_dict[split] = tst_dataset

    return data_dict['train'], data_dict['dev'], data_dict['test']


def make_question_answering_reward(config: "DictConfig") -> QuestionAnsweringReward:
    """
    TODO
    """
    return QuestionAnsweringReward(config.task_lm,
                                   config.task_top_k,
                                   config.num_samples,
                                   config.num_bootstraps,
                                   config.compute_zscore,
                                   config.template,
                                   config.pad_token,
                                   config.lower_outputs,
                                   config.control_output_length,
                                   config.end_punct)
