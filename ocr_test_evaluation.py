#!/usr/bin/env python3
"""
OCR Test Evaluation Script
Uses example_custom_dataset with Qwen3-VL-2B-Instruct model
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import argparse

# Set GPU 4 only
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class OCRTestEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "auto", 
                 prompt: str = None, case_sensitive: bool = False, 
                 ignore_punctuation: bool = True, ignore_spaces: bool = True):
        """
        Initialize OCR Test Evaluator
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cpu", "cuda")
            prompt: Custom prompt for text recognition
            case_sensitive: Whether to preserve case in evaluation
            ignore_punctuation: Whether to ignore punctuation in evaluation
            ignore_spaces: Whether to ignore spaces in evaluation
        """
        print(f"Loading model: {model_name}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Set custom prompt or use default
        self.prompt = prompt if prompt else "Waht is the main word in the image? Output only the text."
        self.case_sensitive = case_sensitive
        self.ignore_punctuation = ignore_punctuation
        self.ignore_spaces = ignore_spaces
        self.model_name = model_name
        
        print(f"Using prompt: {self.prompt}")
        print(f"Case sensitive: {self.case_sensitive}")
        print(f"Ignore punctuation: {self.ignore_punctuation}")
        print(f"Ignore spaces: {self.ignore_spaces}")
        print("Model loaded successfully!")
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load dataset from labels.json
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of dataset items
        """
        labels_path = os.path.join(dataset_path, "labels.json")
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {labels_path}")
        return data
    
    def preprocess_text(self, text: str, case_sensitive: bool = False, 
                       ignore_punctuation: bool = True, ignore_spaces: bool = True) -> str:
        """
        Preprocess text for evaluation
        
        Args:
            text: Raw text
            case_sensitive: Whether to preserve case
            ignore_punctuation: Whether to remove punctuation
            ignore_spaces: Whether to remove spaces
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not case_sensitive:
            text = text.upper()
        
        if ignore_punctuation:
            # Remove punctuation but keep alphanumeric characters
            text = re.sub(r'[^\w\s]', '', text)
        
        if ignore_spaces:
            # Remove all spaces
            text = re.sub(r'\s+', '', text)
        
        return text
    
    def predict_text(self, image_path: str) -> str:
        """
        Predict text from image using Qwen3-VL
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted text
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode prediction
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    def calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate exact match accuracy
        
        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            if pred == gt:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = None) -> Dict:
        """
        Evaluate on the dataset
        
        Args:
            dataset_path: Path to dataset directory
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        predictions = []
        ground_truths = []
        samples = []  # Store original texts for display
        
        print(f"Evaluating {len(dataset)} samples...")
        
        for i, item in enumerate(dataset):
            image_filename = item['image_filename']
            image_path = os.path.join(dataset_path, image_filename)
            
            # Keep original ground truth for display
            original_gt = item['text']
            
            print(f"Processing {i+1}/{len(dataset)}: {image_filename}")
            
            # Predict text
            predicted_text = self.predict_text(image_path)
            
            # Preprocess both for comparison
            processed_gt = self.preprocess_text(original_gt, self.case_sensitive, 
                                               self.ignore_punctuation, self.ignore_spaces)
            processed_pred = self.preprocess_text(predicted_text, self.case_sensitive, 
                                                self.ignore_punctuation, self.ignore_spaces)
            
            predictions.append(processed_pred)
            ground_truths.append(processed_gt)
            
            # Store original texts for display
            samples.append({
                'image_filename': image_filename,
                'image_id': item.get('image_id', i + 1),
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
            'predictions': predictions,
            'ground_truths': ground_truths,
            'samples': samples
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Print evaluation results
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "="*50)
        print("OCR TEST EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*50)
    
    def save_detailed_results(self, results: Dict, output_file: str):
        """
        Save detailed results to text file
        
        Args:
            results: Results dictionary from evaluate_dataset
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*100 + "\n")
            f.write("OCR Test Evaluation Results\n")
            f.write("="*100 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompt: {self.prompt}\n")
            f.write(f"Matching - Case-sensitive: {self.case_sensitive}, Ignore punct: {self.ignore_punctuation}, Ignore space: {self.ignore_spaces}\n")
            f.write("="*100 + "\n\n")
            
            # Sample results
            for i, sample in enumerate(results['samples']):
                f.write(f"Sample {i+1}/{results['total_samples']}\n")
                f.write("-"*100 + "\n")
                f.write(f"Image:          {sample['image_filename']}\n")
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
        
        # Show some examples
        print("\nSample Results:")
        for i, sample in enumerate(results['samples'][:10]):
            status = "✓" if sample['correct'] else "✗"
            print(f"{i+1:2d}. {status} GT: '{sample['ground_truth']}' | Pred: '{sample['predicted_text']}'")


def main():
    parser = argparse.ArgumentParser(description='OCR Test Evaluation with Qwen3-VL')
    parser.add_argument('--dataset_path', type=str, default='example_custom_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-2B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--prompt', type=str, default="Read the main word in the image? Output only the text.",
                       help='Custom prompt for text recognition')
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable case-sensitive evaluation (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore punctuation in evaluation (default: True)')
    parser.add_argument('--ignore-spaces', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore spaces in evaluation (default: True)')
    
    args = parser.parse_args()
    
    # Print evaluation header
    print("Starting OCR Test Evaluation with Qwen3-VL-2B-Instruct")
    print("Using GPU 4 only")
    print(f"Prompt: {args.prompt}")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist!")
        return
    
    # Initialize evaluator
    evaluator = OCRTestEvaluator(
        model_name=args.model_name, 
        device=args.device, 
        prompt=args.prompt,
        case_sensitive=args.case_sensitive,
        ignore_punctuation=args.ignore_punctuation,
        ignore_spaces=args.ignore_spaces
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset_path, args.max_samples)
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results to text file
    output_file = f"ocr_test_results_{args.model_name.replace('/', '_')}.txt"
    evaluator.save_detailed_results(results, output_file)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
