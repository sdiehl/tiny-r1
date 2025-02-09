from unsloth import FastLanguageModel
from vllm import SamplingParams
from prep_data import SYSTEM_PROMPT

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
