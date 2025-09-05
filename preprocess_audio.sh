#!/bin/bash

# Set common paths
WAV_FOLDER_PATH="./VGGSound/audios"
SAVE_FOLDER_PATH="./output"

echo "Starting audio preprocessing..."
echo "WAV folder: $WAV_FOLDER_PATH"
echo "Save folder: $SAVE_FOLDER_PATH"

# Extract mel spectrograms
echo "Extracting mel spectrograms..."
CUDA_VISIBLE_DEVICES=7 python preprocess/extract_mel.py \
    --wav_folder_path $WAV_FOLDER_PATH \
    --save_folder_path $SAVE_FOLDER_PATH

# Extract fbank features
echo "Extracting fbank features..."
CUDA_VISIBLE_DEVICES=7 python preprocess/extract_fbank.py \
    --wav_folder_path $WAV_FOLDER_PATH \
    --save_folder_path $SAVE_FOLDER_PATH

echo "Audio preprocessing completed!"
