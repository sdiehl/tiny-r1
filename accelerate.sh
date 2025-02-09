#!/usr/bin/env bash

set -e

nohup accelerate launch \
  --num_processes 3 \
  --config_file configs/accelerate_configs/deepspeed_zero3.yaml train_trl.py \
  --config zero3.yaml
