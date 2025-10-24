#!/usr/bin/env python3
"""
STR Benchmark Evaluation Script
Evaluates multiple datasets with a single model load
"""

import os
import sys
import argparse
from str_evaluation import STREvaluator

def main():
    parser = argparse.ArgumentParser(description='STR Benchmark Evaluation for multiple datasets')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path to LMDB datasets directory')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='List of dataset names to evaluate')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                       help='Model name to use for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate per dataset (None for all)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default="auto",
                       help='Device to use for inference')
    parser.add_argument('--prompt', type=str, default="What is the main word in the image? Output only the text.",
                       help='Custom prompt for text recognition')
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable case-sensitive evaluation (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore punctuation in evaluation (default: True)')
    parser.add_argument('--ignore-spaces', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore spaces in evaluation (default: True)')
    parser.add_argument('--results_dir', type=str, default="str_benchmark_results",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Print benchmark header
    print("Starting STR Benchmark Evaluation")
    print(f"Model: {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print(f"Case sensitive: {args.case_sensitive}")
    print(f"Ignore punctuation: {args.ignore_punctuation}")
    print(f"Ignore spaces: {args.ignore_spaces}")
    print(f"Max samples per dataset: {args.max_samples}")
    print("=" * 50)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize evaluator (model loaded once)
    print("Loading model...")
    evaluator = STREvaluator(
        model_name=args.model_name,
        device=args.device,
        prompt=args.prompt,
        case_sensitive=args.case_sensitive,
        ignore_punctuation=args.ignore_punctuation,
        ignore_spaces=args.ignore_spaces,
        batch_size=args.batch_size
    )
    print("Model loaded successfully!")
    print("=" * 50)
    
    # Initialize summary data
    summary_data = []
    total_correct = 0
    total_samples = 0
    
    # Evaluate each dataset
    for dataset in args.datasets:
        lmdb_path = os.path.join(args.base_path, dataset)
        
        if not os.path.exists(lmdb_path):
            print(f"Warning: {lmdb_path} not found, skipping...")
            continue
        
        print(f"Evaluating {dataset}...")
        print("-" * 40)
        
        # Run evaluation
        results = evaluator.run_evaluation(
            lmdb_path=lmdb_path,
            max_samples=args.max_samples,
            output_file=os.path.join(args.results_dir, f"{dataset}_results.txt"),
            print_header=False
        )
        
        # Collect summary data
        correct = results['correct_predictions']
        samples = results['total_samples']
        accuracy = results['accuracy']
        
        summary_data.append({
            'dataset': dataset,
            'correct': correct,
            'total': samples,
            'accuracy': accuracy
        })
        
        total_correct += correct
        total_samples += samples
        
        print(f"✓ {dataset}: {correct}/{samples} = {accuracy*100:.2f}%")
        print()
    
    # Generate summary file
    summary_file = os.path.join(args.results_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("STR OCR Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Case Sensitive: {args.case_sensitive}\n")
        f.write(f"Ignore Punctuation: {args.ignore_punctuation}\n")
        f.write(f"Ignore Spaces: {args.ignore_spaces}\n")
        f.write(f"Save Images: False\n\n")
        
        f.write("Results by Dataset:\n")
        f.write("-" * 30 + "\n")
        
        for data in summary_data:
            f.write(f"{data['dataset']:<15}: {data['correct']:>4}/{data['total']:>4} = {data['accuracy']*100:>6.2f}%\n")
        
        f.write("-" * 30 + "\n")
        f.write(f"OVERALL         : {total_correct:>4}/{total_samples:>4} = {total_correct/total_samples*100:>6.2f}%\n\n")
        
        f.write(f"Total Examples: {total_samples}\n")
        f.write(f"Total Correct: {total_correct}\n")
        f.write(f"Overall Accuracy: {total_correct/total_samples*100:.2f}%\n")
    
    print("=" * 50)
    print("STR BENCHMARK EVALUATION COMPLETED")
    print("=" * 50)
    print(f"Total datasets evaluated: {len(summary_data)}")
    print(f"Total samples: {total_samples}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {total_correct/total_samples*100:.2f}%")
    print(f"Results saved in: {args.results_dir}/")
    print(f"Summary file: {summary_file}")

if __name__ == "__main__":
    main()
