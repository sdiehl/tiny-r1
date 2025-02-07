#!/usr/bin/env python

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from prep_data import get_gsm8k_questions
from reward import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
)

def main(training_args, model_args):
    data = get_gsm8k_questions()

    reward_funcs = [
        correctness_reward_func,
        int_reward_func,
        strict_format_reward_func,
        soft_format_reward_func,
        xmlcount_reward_func,
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=data,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)
