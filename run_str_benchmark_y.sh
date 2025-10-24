#!/bin/bash
# Script to evaluate STR LMDB benchmarks with Qwen3-VL
export STR_DATA_DIR=~/data/STR/english_case-sensitive/lmdb/evaluation

# Default: process all samples per dataset with optimized prompt
# Options:
#   --num_samples N           : Number of samples per dataset (default: -1 for all)
#   --batch_size N            : Batch size for inference (default: 8, faster!)
#   --datasets "CUTE80,SVT"   : Specific datasets to evaluate
#   --prompt "text"           : Custom prompt
#   --case-sensitive false    : Case-insensitive matching (default: false)
#   --ignore-punctuation true : Ignore punctuation (default: true)
#   --ignore-space true       : Ignore spaces (default: true)
#   --model MODEL             : Different model
#   --results_dir DIR         : Results directory (default: str_benchmark_results)

python str_evaluation.py \
    --datasets "CUTE80,SVT,SVTP,IC13_857,IC15_1811,IIIT5k_3000" \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --max_samples -1 \
    --batch_size 1 \
    --device auto \
    --prompt "What is the main word in the image? Output only the text." \
    --case-sensitive false \
    --ignore-punctuation true \
    --ignore-spaces true \
    --results_dir "str_benchmark_results" \
    "$@"