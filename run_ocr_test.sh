#!/bin/bash

# OCR Test Evaluation Script
# Edit the configuration below to change evaluation settings

# ===== CONFIGURATION =====
# Edit the values below directly on each line
# =========================

# Set GPU 4 only
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

# Run evaluation
python ocr_test_evaluation.py \
    --dataset_path example_custom_dataset \
    --model_name "Qwen/Qwen3-VL-2B-Instruct" \
    --max_samples 10 \
    --device auto \
    --prompt "What is the main word in the image? Output only the text." \
    --case-sensitive false \
    --ignore-punctuation true \
    --ignore-spaces true

echo "OCR Test Evaluation completed!"
