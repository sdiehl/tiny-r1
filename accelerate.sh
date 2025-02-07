#!/usr/bin/env bash

set -e

nohup accelerate launch \
    --num_processes 3 \
    --config_file zero3.yaml train_trl.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/Llama-3.1-8B-Reasoning \
    --max_prompt_length 2048 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 20 \
    --learning_rate 3e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --num_generations 3 \
    --save_steps 50 \
    --max_steps 1000 \
    --torch_dtype bfloat16 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --bf16