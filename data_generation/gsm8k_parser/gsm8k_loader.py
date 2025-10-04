"""
GSM8K dataset loader and preprocessor for ACOR system
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from datasets import load_dataset
import pandas as pd


class GSM8KLoader:
    """Loader and preprocessor for GSM8K dataset"""

    def __init__(self):
        self.logger = logging.getLogger("acor.data_generation.gsm8k_loader")
        self.dataset = None

    def load_dataset(self, split: str = "train") -> List[Dict[str, str]]:
        """
        Load GSM8K dataset from HuggingFace

        Args:
            split: Dataset split to load ('train' or 'test')

        Returns:
            List of examples with 'question' and 'answer' fields
        """
        self.logger.info(f"Loading GSM8K {split} dataset...")

        try:
            self.dataset = load_dataset("gsm8k", "main", split=split)
            examples = []

            for item in self.dataset:
                examples.append({
                    'question': item['question'],
                    'answer': item['answer']
                })

            self.logger.info(f"Loaded {len(examples)} examples from GSM8K {split}")
            return examples

        except Exception as e:
            self.logger.error(f"Failed to load GSM8K dataset: {e}")
            return []

    def clean_answer(self, answer: str) -> Tuple[str, Optional[float]]:
        """
        Clean and parse GSM8K answer format

        Args:
            answer: Raw answer string from GSM8K

        Returns:
            Tuple of (cleaned_solution, final_numeric_answer)
        """
        # GSM8K answers end with "#### final_number"
        if "####" in answer:
            solution, final_answer = answer.rsplit("####", 1)
            solution = solution.strip()
            final_answer = final_answer.strip()

            # Try to extract numeric answer
            numeric_answer = None
            try:
                # Remove commas and extract number
                clean_number = re.sub(r'[,$]', '', final_answer)
                numeric_answer = float(clean_number)
            except ValueError:
                self.logger.warning(f"Could not parse final answer: {final_answer}")

            return solution, numeric_answer
        else:
            # No clear final answer marker
            return answer, None

    def extract_solution_steps(self, solution: str) -> List[str]:
        """
        Extract individual reasoning steps from solution

        Args:
            solution: Cleaned solution text

        Returns:
            List of reasoning steps
        """
        steps = []

        # Split by sentence patterns
        sentences = re.split(r'[.!?]+', solution)

        current_step = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if this starts a new step (contains numbers/calculations)
            has_calculation = bool(re.search(r'\d+.*[+\-*/×÷=].*\d+', sentence))
            has_step_indicator = any(indicator in sentence.lower() for indicator in [
                'first', 'then', 'next', 'so', 'therefore', 'now', 'finally'
            ])

            if (has_calculation or has_step_indicator) and current_step:
                # Save previous step and start new one
                steps.append(current_step.strip())
                current_step = sentence
            else:
                # Continue current step
                if current_step:
                    current_step += " " + sentence
                else:
                    current_step = sentence

        # Add final step if exists
        if current_step:
            steps.append(current_step.strip())

        return [step for step in steps if len(step) > 10]  # Filter very short steps

    def validate_example(self, example: Dict[str, str]) -> bool:
        """
        Validate a GSM8K example

        Args:
            example: Dictionary with 'question' and 'answer' fields

        Returns:
            True if valid, False otherwise
        """
        if not example.get('question') or not example.get('answer'):
            return False

        question = example['question']
        answer = example['answer']

        # Check reasonable lengths
        if len(question) < 20 or len(question) > 500:
            return False

        if len(answer) < 20 or len(answer) > 1000:
            return False

        # Check for mathematical content
        if not re.search(r'\d+', question):
            return False

        # Check for final answer format
        if "####" not in answer:
            return False

        return True

    def preprocess_for_training(
        self,
        examples: List[Dict[str, str]],
        max_examples: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Preprocess GSM8K examples for training data generation

        Args:
            examples: Raw GSM8K examples
            max_examples: Maximum number of examples to process

        Returns:
            List of preprocessed examples
        """
        if max_examples:
            examples = examples[:max_examples]

        processed = []

        for example in examples:
            if not self.validate_example(example):
                continue

            # Clean the answer
            solution, final_answer = self.clean_answer(example['answer'])

            # Extract reasoning steps
            steps = self.extract_solution_steps(solution)

            processed_example = {
                'question': example['question'],
                'original_answer': example['answer'],
                'solution': solution,
                'final_numeric_answer': final_answer,
                'reasoning_steps': steps,
                'num_steps': len(steps)
            }

            processed.append(processed_example)

        self.logger.info(f"Preprocessed {len(processed)}/{len(examples)} valid examples")
        return processed

    def get_problem_types(self, examples: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Analyze problem types in the dataset

        Args:
            examples: List of GSM8K examples

        Returns:
            Dictionary with problem type counts
        """
        type_patterns = {
            'arithmetic': [r'add', r'sum', r'total', r'plus', r'minus', r'subtract'],
            'multiplication': [r'times', r'multiply', r'product', r'each.*\d+'],
            'division': [r'divide', r'split', r'share', r'per', r'average'],
            'percentage': [r'percent', r'%', r'discount', r'tax'],
            'fractions': [r'1/2', r'1/3', r'1/4', r'half', r'third', r'quarter'],
            'word_problems': [r'john', r'mary', r'sarah', r'store', r'school'],
            'geometry': [r'area', r'perimeter', r'rectangle', r'circle', r'square'],
            'time': [r'hour', r'minute', r'day', r'week', r'month', r'year'],
            'money': [r'dollar', r'\$', r'cost', r'price', r'buy', r'sell', r'profit']
        }

        type_counts = {ptype: 0 for ptype in type_patterns.keys()}

        for example in examples:
            question_lower = example['question'].lower()

            for ptype, patterns in type_patterns.items():
                if any(re.search(pattern, question_lower) for pattern in patterns):
                    type_counts[ptype] += 1

        return type_counts

    def save_preprocessed(
        self,
        processed_examples: List[Dict[str, str]],
        output_path: Path
    ) -> bool:
        """
        Save preprocessed examples to file

        Args:
            processed_examples: List of preprocessed examples
            output_path: Path to save the data

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for example in processed_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')

            self.logger.info(f"Saved {len(processed_examples)} preprocessed examples to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save preprocessed data: {e}")
            return False

    def create_sample_for_testing(
        self,
        examples: List[Dict[str, str]],
        sample_size: int = 10
    ) -> List[Dict[str, str]]:
        """
        Create a small sample for testing

        Args:
            examples: Full list of examples
            sample_size: Number of examples in sample

        Returns:
            List of sample examples
        """
        # Select diverse examples
        sample = []
        step = max(1, len(examples) // sample_size)

        for i in range(0, len(examples), step):
            if len(sample) >= sample_size:
                break
            sample.append(examples[i])

        return sample