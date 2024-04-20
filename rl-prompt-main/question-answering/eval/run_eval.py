import sys
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline

sys.path.append('..')  # A hack
import json
import pandas as pd
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from qa_evaluator import QuestionAnsweringEvaluator
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from qa_helpers import (QuestionAnsweringDatasetConfig,
                        QuestionAnsweringRewardConfig,
                        load_question_answering_dataset, make_question_answering_datasets)
import hydra
from dataclasses import dataclass


@dataclass
class QuestionAnsweringEvaluationConfig:
            task_lm: str = "distilgpt2"
            prompt: Optional[str] = None
            pad_token: str = '<|endoftext|>'
            lower_outputs: bool = False  # Whether to convert all outputs to lower case
            control_output_length: bool = False
            end_punct: str = '"'


# Compose default config
config_list = [QuestionAnsweringDatasetConfig,
               QuestionAnsweringEvaluationConfig]
cs = compose_hydra_config_store('base_eval', config_list)

ppl_lm_dict = {'squad': './ppl/gpt2-yelp',
               'shakespeare': './ppl/gpt2-shakespeare'}
@hydra.main(version_base=None, config_path="./", config_name="eval_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')

    if config.prompt is None:
        raise ValueError('Need to supply at least one prompt')

    output_dir = get_hydra_output_dir()

    _, _, test_dataset = make_question_answering_datasets(config)
    print('Test Size:', len(test_dataset))
    print('Examples:', test_dataset[:5])

    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1)

    all_source_texts = []
    all_target_labels = []
    all_ref_texts = []
    all_output_texts = []
    all_rewards = []

    evaluator = QuestionAnsweringEvaluator(config.task_lm,
                                           config.prompt,
                                           config.pad_token,
                                           config.lower_outputs,
                                           config.control_output_length,
                                           config.end_punct)

    questions, answers, gts, em_scores, f1_scores = evaluator.evaluate_output(test_loader)

    output_data = {'prompt': config.prompt,
                   'questions': questions,
                   'gts': gts, 'answers': answers,
                   'em_scores': em_scores, 'f1_scores': f1_scores}
    output_data_df = pd.DataFrame(output_data)
    # summary_path = os.path.join(output_dir, 'summary.json')
    output_path = os.path.join(output_dir, 'outputs.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    # output_data_df.to_csv(output_path, index=False, sep='\t')
    print(f'Outputs saved at {output_dir}')


if __name__ == "__main__":
    main()
