<p align="center">
  <img src=".github/logo.jpeg" alt="Tiny-R1 Logo" width="400px">
</p>

# Tiny-R1

Recreating the DeepSeek-R1 GRPO reinforcement learning process by fine-tuning a smaller language model to exhibit reasoning behavior.

We process the [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset and train the model. We provide two training approaches:

1. [GPU Rich](#1-multi-gpu-training-acceleratedeepspeed) ( 4 x H200 SXM )
2. [GPU Poor](#2-single-gpu-training-unsloth) ( 1 x A100 SXM )

## Data Processing

Question:

```
Bob gets paid $5 an hour for the regular hours he works and $6 an hour for any overtime hours he works. All hours over 40 in a week are considered overtime. If Bob works 44 hours in the first week and 48 hours in the second week, how much did he make?
```

Answer:

```
Bob has worked 40 hours/week x 2 weeks = 80 regular hours Bob got paid 80 hours x $5/hour = $400 for his regular hours Bob worked 44 hours - 40 hours = 4 hours of overtime in the first week Bob worked 48 hours - 40 hours = 8 hours of overtime in the second week In total Bob worker 8 hours + 4 hours =12 hours of overtime. Bob got paid $6/hour x 12 hours = $72 for overtime In 2 weeks Bob earned $400 + $72 = $472 #### 472
```

Into the format ChatML format enriched with reasoning and answer in XML format.

```xml
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
<|eot_id|><|start_header_id|>user<|end_header_id|>
Bob gets paid $5 an hour for the regular hours he works and $6 an hour for any overtime hours he works. All hours over 40 in a week are considered overtime. If Bob works 44 hours in the first week and 48 hours in the second week, how much did he make?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<reasoning>
Bob has worked 40 hours/week x 2 weeks = 80 regular hours
Bob got paid 80 hours x $5/hour = $400 for his regular hours
Bob worked 44 hours - 40 hours = 4 hours of overtime in the first week
Bob worked 48 hours - 40 hours = 8 hours of overtime in the second week
In total Bob worker 8 hours + 4 hours = 12 hours of overtime.
Bob got paid $6/hour x 12 hours = $72 for overtime
In 2 weeks Bob earned $400 + $72 = $472
</reasoning>
<answer>
472
</answer>
<|eot_id|>
```

## 1. Multi-GPU Training (Accelerate/DeepSpeed)

This approach is suitable for environments with multiple GPUs and uses Accelerate with DeepSpeed ZeRO-3 for distributed training.

```bash
# Install required packages
./setup.sh
```

```bash
# Launch multi-GPU training
./accelerate.sh
```

The multi-GPU configuration uses DeepSpeed ZeRO-3 as defined in `zero3.yaml`:

```zero3.yaml
output_dir: outputs/Llama-3.1-8B-Reasoning
model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
model_revision: main

# Hyperparameters
learning_rate: 5.0e-7
lr_scheduler_type: cosine
max_steps: 450
gradient_accumulation_steps: 8
per_device_train_batch_size: 1
warmup_ratio: 0.03
beta: 0.001
max_prompt_length: 256
max_completion_length: 1024
num_generations: 8
seed: 1337

# Inference Compute
use_vllm: true
vllm_gpu_memory_utilization: 0.5
vllm_device: "cuda:3"

# Memory Usage
torch_dtype: bfloat16
attn_implementation: flash_attention_2
bf16: true
tf32: true
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

# Logging
logging_strategy: steps
logging_steps: 2
report_to:
- wandb

# Saving
save_strategy: "steps"
save_steps: 25
```

## 2. Single-GPU Training (Unsloth)

This approach is optimized for single GPU environments using Unsloth's optimizations.

```bash
# Install required packages
./setup.sh
```

```bash
python train_unsloth.py
```

Key training parameters from `train_unsloth.py`:

```python
training_args = GRPOConfig(
    use_vllm=True,
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
    report_to="wandb",
    output_dir="outputs",
)
```

## Testing the Model

After training, you can test the model using the provided test script:

```python
from unsloth import FastLanguageModel
from vllm import SamplingParams

LORA_NAME = "grpo_saved_lora"
SAMPLE_PROMPT = "Which is greater 9.10 or 9.9?"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_PROMPT},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora(LORA_NAME),
)

answer = output[0].outputs[0].text
print(answer)
```
