#!/usr/bin/env bash

set -e

# Function to install common dependencies
install_common_deps() {
    pip install --upgrade pip
    pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
    pip install flash-attn
    pip install --no-deps peft bitsandbytes
}

# Function to install Unsloth dependencies
install_unsloth() {
    install_common_deps
    pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
    pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
    pip install "vllm==0.7.0"
}

# Function to install Accelerate dependencies
install_accelerate() {
    install_common_deps
    pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
    pip install "deepspeed==0.15.4"
    pip install "accelerate==1.3.0"
    pip install "vllm==0.7.0"
}

# Main script
echo "Please select your training mode:"
echo "1) Single-GPU Training (Unsloth)"
echo "2) Multi-GPU Training (Accelerate/DeepSpeed)"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Installing dependencies for Unsloth mode..."
        install_unsloth
        ;;
    2)
        echo "Installing dependencies for Accelerate/DeepSpeed mode..."
        install_accelerate
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac

echo "Setup completed successfully!"
