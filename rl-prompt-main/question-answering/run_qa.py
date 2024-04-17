import os
import hydra

from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.models import LMAdaptorModelConfig, SinglePromptModelConfig, make_lm_adaptor_model, make_single_prompt_model
from rlprompt.utils.utils import colorful_print, compose_hydra_config_store, get_hydra_output_dir

from qa_helpers import QuestionAnsweringRewardConfig, QuestionAnsweringDatasetConfig, make_question_answering_reward, make_question_answering_datasets

@hydra.main(version_base=None, config_path="./", config_name="qa_config")
def main(config: "DictConfig"):
    """
    
    """
    print("Entered main.")
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = make_question_answering_datasets(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size:', len(val_dataset))
    print('Examples:', val_dataset[:5])

    # TODO: maybe?
    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)

    reward = make_question_answering_reward(config)
    algo_module = make_sql_module(prompt_model, reward, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()