from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from prep_data import get_gsm8k_questions
from reward import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
)

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 512  # Need to increase for longer reasoning
lora_rank = 32

LORA_NAME = "grpo_saved_lora"
RANDOM_STATE = 1337

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,  # Use vLLM for fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_STATE,
)


def main():
    dataset = get_gsm8k_questions()

    training_args = GRPOConfig(
        use_vllm=True,  # Use the vllm for inference
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        # report_to="wandb",
        output_dir="outputs",
    )

    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    print("Training...")
    trainer.train()
    print("Done training.")

    print("Saving...")
    model.save_lora(LORA_NAME)
    model.save_pretrained_merged(
        "model",
        tokenizer,
        save_method="merged_16bit",
    )
    model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    print("Done saving.")


if __name__ == "__main__":
    main()
