pip install --upgrade pip

# Install torch
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn
pip install flash-attn

# Install unsloth
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

# Install TRL
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b

# Install base ML packages
pip install --no-deps peft bitsandbytes

# Install deepspeed
pip install "deepspeed==0.15.4"

# Install accelerate
pip install "accelerate==1.3.0"

# Install vllm
pip install "vllm==0.7.0"

# Update Pillow
pip install --upgrade pillow
