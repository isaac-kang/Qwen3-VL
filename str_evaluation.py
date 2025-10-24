#!/usr/bin/env python3
"""
STR (Scene Text Recognition) Evaluation Script for LMDB datasets
Supports multiple STR benchmark datasets in LMDB format
"""

import os
import re
import lmdb
import json
import pathlib
import typing
import torch
import argparse
from PIL import Image
from io import BytesIO
from transformers import AutoModelForImageTextToText, AutoProcessor

# Set GPU 4 only
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class STREvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "auto",
                 prompt: str = None, case_sensitive: bool = False,
                 ignore_punctuation: bool = True, ignore_spaces: bool = True,
                 batch_size: int = 1):
        self.model_name = model_name
        self.device = device
        self.prompt = prompt or "What is the main word in the image? Output only the text."
        self.case_sensitive = case_sensitive
        self.ignore_punctuation = ignore_punctuation
        self.ignore_spaces = ignore_spaces
        self.batch_size = batch_size
        
        print(f"Loading model: {model_name}")
        print(f"Using prompt: {self.prompt}")
        print(f"Case sensitive: {self.case_sensitive}")
        print(f"Ignore punctuation: {self.ignore_punctuation}")
        print(f"Ignore spaces: {self.ignore_spaces}")
        print(f"Batch size: {self.batch_size}")
        
        try:
            # Load model and processor
            print("Loading model and processor...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            print("Model loaded successfully!")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_lmdb_dataset(self, lmdb_path: str):
        """Load dataset from LMDB format"""
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        dataset = []
        with env.begin(write=False) as txn:
            # Get all image keys and their corresponding labels
            image_data = {}
            label_data = {}
            
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('utf-8')
                
                if key_str.startswith('image-'):
                    # Store image data
                    image_data[key_str] = value
                elif key_str.startswith('label-'):
                    # Store label data
                    label_data[key_str] = value.decode('utf-8')
            
            # Match images with their labels
            for image_key in sorted(image_data.keys()):
                # Extract image ID (e.g., 'image-000000001' -> '000000001')
                image_id = image_key.replace('image-', '')
                label_key = f'label-{image_id}'
                
                if label_key in label_data:
                    # Convert image data to PIL Image
                    image = Image.open(BytesIO(image_data[image_key])).convert('RGB')
                    
                    dataset.append({
                        'image': image,
                        'text': label_data[label_key],
                        'image_id': image_id
                    })
        
        env.close()
        return dataset
    
    def preprocess_text(self, text: str, case_sensitive: bool = False,
                       ignore_punctuation: bool = True, ignore_spaces: bool = True) -> str:
        """Preprocess text for evaluation comparison"""
        processed = text
        
        if not case_sensitive:
            processed = processed.upper()
        
        if ignore_punctuation:
            processed = re.sub(r'[^\w\s]', '', processed)
        
        if ignore_spaces:
            processed = processed.replace(' ', '')
        
        return processed
    
    def predict_text(self, image: Image.Image) -> str:
        """Generate text prediction for given image"""
        # Prepare messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]
        
        # Process inputs using chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Extract only the generated part (remove prompt and chat template artifacts)
        if self.prompt in generated_text:
            generated_text = generated_text.replace(self.prompt, "").strip()
        
        # Remove chat template artifacts
        if "user\n\nassistant\n" in generated_text:
            generated_text = generated_text.split("user\n\nassistant\n")[-1].strip()
        elif "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1].strip()
        
        return generated_text
    
    def predict_batch(self, images: list) -> list:
        """Generate text predictions for multiple images using true batch processing"""
        if not images:
            return []
        
        # For batch generation, padding_side should be set to left!
        self.processor.tokenizer.padding_side = 'left'
        
        # Create separate messages for each image (batch inference)
        messages_batch = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            messages_batch.append(messages)
        
        # Process batch using chat template with padding
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True  # padding should be set for batch generation!
        )
        inputs = inputs.to(self.model.device)
        
        # Generate for batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Trim generated_ids to remove input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode batch results
        generated_texts = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Clean up each result
        results = []
        for generated_text in generated_texts:
            # Remove chat template artifacts
            if "user\n\nassistant\n" in generated_text:
                generated_text = generated_text.split("user\n\nassistant\n")[-1].strip()
            elif "assistant\n" in generated_text:
                generated_text = generated_text.split("assistant\n")[-1].strip()
            
            results.append(generated_text.strip())
        
        # Reset padding_side to right for single inference
        self.processor.tokenizer.padding_side = 'right'
        
        return results
    
    def evaluate_dataset(self, lmdb_path: str, max_samples: int = None):
        """Evaluate dataset and return results"""
        print(f"Loading LMDB dataset from: {lmdb_path}")
        dataset = self.load_lmdb_dataset(lmdb_path)
        
        # Handle max_samples=-1 for full dataset
        if max_samples is not None and max_samples != -1:
            dataset = dataset[:max_samples]
        elif max_samples == -1:
            print(f"Evaluating full dataset: {len(dataset)} samples")
        
        predictions = []
        ground_truths = []
        samples = []  # Store original texts for display
        
        print(f"Evaluating {len(dataset)} samples...")
        
        # Process in batches
        for batch_start in range(0, len(dataset), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1}/{(len(dataset) + self.batch_size - 1)//self.batch_size}: samples {batch_start+1}-{batch_end}")
            
            # Extract images and metadata
            batch_images = [item['image'] for item in batch]
            batch_gt = [item['text'] for item in batch]
            batch_ids = [item['image_id'] for item in batch]
            
            # Predict batch
            if self.batch_size == 1:
                # Single image processing
                predicted_texts = [self.predict_text(img) for img in batch_images]
            else:
                # Batch processing
                predicted_texts = self.predict_batch(batch_images)
            
            # Process each item in the batch
            for i, (item, predicted_text) in enumerate(zip(batch, predicted_texts)):
                original_gt = item['text']
                image_id = item['image_id']
                
                # Preprocess both for comparison
                processed_gt = self.preprocess_text(original_gt, self.case_sensitive, 
                                                   self.ignore_punctuation, self.ignore_spaces)
                processed_pred = self.preprocess_text(predicted_text, self.case_sensitive, 
                                                    self.ignore_punctuation, self.ignore_spaces)
                
                predictions.append(processed_pred)
                ground_truths.append(processed_gt)
                
                # Store original texts for display
                samples.append({
                    'image_id': image_id,
                    'predicted_text': predicted_text,  # Original prediction
                    'ground_truth': original_gt,  # Original ground truth
                    'correct': processed_pred == processed_gt
                })
                
                # Print result with original values for display
                status = "✓" if processed_pred == processed_gt else "✗"
                print(f"  {status} GT: '{original_gt}' | Pred: '{predicted_text}'")
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(predictions, ground_truths)
        
        # Count correct predictions
        correct_count = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        
        results = {
            'total_samples': len(dataset),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'samples': samples
        }
        
        return results
    
    def calculate_accuracy(self, predictions: list, ground_truths: list) -> float:
        """Calculate accuracy between predictions and ground truths"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        return correct / len(predictions)
    
    def print_results(self, results: dict):
        """Print evaluation results"""
        print("="*50)
        print("STR EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*50)
        
        # Show sample results
        print("Sample Results:")
        for i, sample in enumerate(results['samples'][:10]):  # Show first 10
            status = "✓" if sample['correct'] else "✗"
            print(f"{i+1:2d}. {status} GT: '{sample['ground_truth']}' | Pred: '{sample['predicted_text']}'")
    
    def save_detailed_results(self, results: dict, output_file: str):
        """Save detailed results to text file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*100 + "\n")
            f.write("STR Evaluation Results\n")
            f.write("="*100 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompt: {self.prompt}\n")
            f.write(f"Matching - Case-sensitive: {self.case_sensitive}, Ignore punct: {self.ignore_punctuation}, Ignore space: {self.ignore_spaces}\n")
            f.write("="*100 + "\n\n")
            
            # Sample results
            for i, sample in enumerate(results['samples']):
                f.write(f"Sample {i+1}/{results['total_samples']}\n")
                f.write("-"*100 + "\n")
                f.write(f"Image ID:       {sample['image_id']}\n")
                f.write(f"Prompt:         {self.prompt}\n")
                f.write(f"Model Answer:   {sample['predicted_text']}\n")
                f.write(f"Ground Truth:   {sample['ground_truth']}\n")
                f.write(f"Correct:        {'✓' if sample['correct'] else '✗'}\n")
                f.write("\n\n")
            
            # Summary
            f.write("="*100 + "\n")
            f.write(f"Dataset Complete!\n")
            f.write(f"Accuracy: {results['correct_predictions']}/{results['total_samples']} = {results['accuracy']*100:.2f}%\n")
            f.write("="*100 + "\n")
    
    
    def run_evaluation(self, lmdb_path: str, max_samples: int = None, output_file: str = None, print_header: bool = True):
        """Run complete evaluation pipeline"""
        if print_header:
            print(f"Evaluating dataset: {os.path.basename(lmdb_path)}")
            print("-" * 40)
        
        results = self.evaluate_dataset(lmdb_path, max_samples)
        
        if print_header:
            self.print_results(results)
        
        if output_file:
            self.save_detailed_results(results, output_file)
            if print_header:
                print(f"Detailed results saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='STR Evaluation for LMDB datasets')
    parser.add_argument('--lmdb_path', type=str, default=None,
                       help='Path to LMDB dataset directory (single dataset)')
    parser.add_argument('--lmdb_paths', nargs='+', default=None,
                       help='List of LMDB dataset paths (multiple datasets)')
    parser.add_argument('--datasets', type=str, default='CUTE80,SVT,SVTP,IC13_857,IC15_1811,IIIT5k_3000',
                       help='Comma-separated list of datasets to evaluate')
    parser.add_argument('--base_path', type=str, default=None,
                       help='Base path to LMDB datasets directory (default: $STR_DATA_DIR)')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                       help='Model name to use for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
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
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for detailed results')
    parser.add_argument('--results_dir', type=str, default="str_results",
                       help='Results directory for multiple datasets')
    parser.add_argument('--summary_file', type=str, default=None,
                       help='Summary file for multiple datasets')
    
    args = parser.parse_args()
    
    # Set base_path from environment variable if not provided
    if args.base_path is None:
        if 'STR_DATA_DIR' not in os.environ:
            print("Error: STR_DATA_DIR environment variable is not set!")
            print("Please set STR_DATA_DIR to the path of your STR datasets directory.")
            print("Example: export STR_DATA_DIR=/path/to/STR/datasets")
            return
        args.base_path = os.environ['STR_DATA_DIR']
    
    # Determine if single or multiple datasets
    if args.lmdb_paths:
        # Multiple datasets mode
        lmdb_paths = args.lmdb_paths
    elif args.datasets:
        # Multiple datasets mode using --datasets option
        dataset_names = [d.strip() for d in args.datasets.split(',')]
        lmdb_paths = [os.path.join(args.base_path, dataset) for dataset in dataset_names]
    else:
        lmdb_paths = None
    
    if lmdb_paths:
        print("Starting STR Benchmark Evaluation")
        print(f"Model: {args.model_name}")
        print(f"Prompt: {args.prompt}")
        print(f"Case sensitive: {args.case_sensitive}")
        print(f"Ignore punctuation: {args.ignore_punctuation}")
        print(f"Ignore spaces: {args.ignore_spaces}")
        print(f"Max samples per dataset: {args.max_samples}")
        print("="*50)
        
        # Create results directory
        os.makedirs(args.results_dir, exist_ok=True)
        
        # Create evaluator (model loaded once)
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
        print("="*50)
        
        # Initialize summary data
        summary_data = []
        total_correct = 0
        total_samples = 0
        
        # Evaluate each dataset
        for lmdb_path in lmdb_paths:
            if not os.path.exists(lmdb_path):
                print(f"Warning: {lmdb_path} not found, skipping...")
                continue
            
            dataset_name = os.path.basename(lmdb_path)
            print(f"Evaluating {dataset_name}...")
            print("-" * 40)
            
            # Run evaluation
            results = evaluator.run_evaluation(
                lmdb_path=lmdb_path,
                max_samples=args.max_samples,
                output_file=os.path.join(args.results_dir, f"{dataset_name}_results.txt"),
                print_header=False
            )
            
            # Collect summary data
            correct = results['correct_predictions']
            samples = results['total_samples']
            accuracy = results['accuracy']
            
            summary_data.append({
                'dataset': dataset_name,
                'correct': correct,
                'total': samples,
                'accuracy': accuracy
            })
            
            total_correct += correct
            total_samples += samples
            
            print(f"✓ {dataset_name}: {correct}/{samples} = {accuracy*100:.2f}%")
            print()
        
        # Generate summary file
        if args.summary_file:
            summary_file = args.summary_file
        else:
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
        
    else:
        # Single dataset mode (original behavior)
        if not args.lmdb_path:
            print("Error: Either --lmdb_path or --lmdb_paths must be specified")
            return
            
        # Create evaluator
        evaluator = STREvaluator(
            model_name=args.model_name,
            device=args.device,
            prompt=args.prompt,
            case_sensitive=args.case_sensitive,
            ignore_punctuation=args.ignore_punctuation,
            ignore_spaces=args.ignore_spaces,
            batch_size=args.batch_size
        )
        
        # Generate output filename if not provided
        if not args.output_file:
            dataset_name = os.path.basename(args.lmdb_path)
            model_name_clean = args.model_name.replace('/', '_')
            args.output_file = f"str_results_{dataset_name}_{model_name_clean}.txt"
        
        # Run evaluation
        results = evaluator.run_evaluation(
            lmdb_path=args.lmdb_path,
            max_samples=args.max_samples,
            output_file=args.output_file
        )


if __name__ == "__main__":
    main()
