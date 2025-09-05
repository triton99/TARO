#!/bin/bash

# Set common paths
VIDEO_FOLDER_PATH="./VGGSound/videos"
SAVE_FOLDER_PATH="./output"

echo "Starting video preprocessing..."
echo "Video folder: $VIDEO_FOLDER_PATH"
echo "Save folder: $SAVE_FOLDER_PATH"

# Extract CAVP features
echo "Extracting CAVP features..."
CUDA_VISIBLE_DEVICES=0 python preprocess/extract_cavp.py \
    --video_folder_path $VIDEO_FOLDER_PATH \
    --save_folder_path $SAVE_FOLDER_PATH \
    --cavp_config_path ./cavp/cavp.yaml \
    --cavp_ckpt_path ./ckpts/cavp_epoch66.ckpt

# Extract onset features
echo "Extracting onset features..."
CUDA_VISIBLE_DEVICES=0 python preprocess/extract_onset.py \
    --video_folder_path $VIDEO_FOLDER_PATH \
    --save_folder_path $SAVE_FOLDER_PATH \
    --onset_ckpt_path ./ckpts/onset_model.ckpt

echo "Video preprocessing completed!"
