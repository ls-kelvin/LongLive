#!/bin/bash

# Video-conditional inference script for LongLive model
# This script supports both single-GPU and multi-GPU inference

# Set the config file path
CONFIG_PATH="configs/longlive_video_conditional_inference.yaml"

# Check if running in multi-GPU mode
if [ -z "$WORLD_SIZE" ]; then
    # Single GPU mode
    echo "Running in single-GPU mode..."
    python video_conditional_inference.py --config_path $CONFIG_PATH
else
    # Multi-GPU mode (launched via torchrun)
    echo "Running in multi-GPU mode with $WORLD_SIZE GPUs..."
    python video_conditional_inference.py --config_path $CONFIG_PATH
fi
