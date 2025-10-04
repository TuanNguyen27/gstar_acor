"""
Generate training data for the Planner module from GSM8K
"""
import json
from typing import Dict, List, Any
from pathlib import Path

from ..base_generator import BaseDataGenerator


class PlannerDataGenerator(BaseDataGenerator):
    """Generate (question, plan) pairs for Planner module training"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _validate_example_impl(self, example: Dict[str, Any]) -> bool:
        """Validate a planner training example"""
        required_fields = ['question', 'plan', 'steps']

        # Check required fields
        for field in required_fields:
            if field not in example:
                return False

        # Check content quality
        question = example['question']
        plan = example['plan']
        steps = example['steps']

        # Question validation
        if not question or len(question) < 20:
            return False

        # Plan validation
        if not plan or len(plan) < 50:
            return False

        # Steps validation
        if not isinstance(steps, list) or len(steps) < 2 or len(steps) > 8:
            return False

        # Check that steps are reasonably detailed
        for step in steps:
            if not step.get('description') or len(step['description']) < 10:
                return False

        return True

    def create_plan_extraction_prompt(self, question: str, solution: str) -> List[Dict[str, str]]:
        """Create prompt for extracting plan from GSM8K solution"""
        return [
            {
                "role": "system",
                "content": """You are a strategic planner for mathematical problem solving.
Your task is to extract a high-level plan from a complete solution.

Guidelines:
1. Create 3-6 clear, sequential steps that form a strategic blueprint
2. Each step should describe WHAT to do, not HOW to do the detailed calculations
3. Focus on the logical flow and problem-solving strategy
4. Number each step clearly
5. Keep steps concise but descriptive
6. Don't include the actual calculations, just the strategic actions

Example:
Problem: "Tom has 3 bags with 15 marbles each. He gives away 12 marbles. How many marbles does he have left?"
Solution: "Tom has 3 bags with 15 marbles each, so he has 3 * 15 = 45 marbles initially. After giving away 12 marbles, he has 45 - 12 = 33 marbles left. #### 33"

Plan:
1. Calculate the total number of marbles across all bags
2. Subtract the number of marbles given away
3. Determine the final count of remaining marbles"""
            },
            {
                "role": "user",
                "content": f"""Problem: {question}
Solution: {solution}

Extract a strategic plan from this solution:"""
            }
        ]

    def generate_examples(
        self,
        input_data: List[Dict[str, Any]],
        num_examples: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate planner training examples from GSM8K data

        Args:
            input_data: List of preprocessed GSM8K examples
            num_examples: Number of examples to generate

        Returns:
            List of planner training examples
        """
        examples = []

        for i, gsm8k_example in enumerate(input_data):
            if len(examples) >= num_examples:
                break

            question = gsm8k_example['question']
            solution = gsm8k_example['solution']

            # Generate plan using teacher model
            messages = self.create_plan_extraction_prompt(question, solution)
            plan_response = self.call_teacher_model(messages, max_tokens=300)

            if not plan_response:
                continue

            # Parse the plan into structured steps
            steps = self.parse_plan_steps(plan_response)

            if len(steps) < 2:  # Need at least 2 steps for valid plan
                continue

            training_example = {
                'question': question,
                'plan': plan_response,
                'steps': steps,
                'original_solution': solution,
                'source': 'gsm8k_generated'
            }

            examples.append(training_example)

            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {len(examples)} planner examples from {i + 1} GSM8K problems")

        return examples

    def parse_plan_steps(self, plan_text: str) -> List[Dict[str, Any]]:
        """Parse plan text into structured steps"""
        import re

        steps = []

        # Look for numbered patterns
        step_patterns = [
            r'(\d+)\.\s*(.+?)(?=\n\d+\.|\n[A-Z]|\Z)',  # "1. step text"
            r'(\d+)\)\s*(.+?)(?=\n\d+\)|\n[A-Z]|\Z)',  # "1) step text"
            r'Step\s+(\d+):?\s*(.+?)(?=\nStep|\n[A-Z]|\Z)',  # "Step 1: text"
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, plan_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    step_num = int(match[0])
                    step_text = match[1].strip().replace('\n', ' ')

                    if len(step_text) > 5:  # Minimum step length
                        steps.append({
                            'step_number': step_num,
                            'description': step_text
                        })
                break

        # Fallback parsing
        if not steps:
            lines = plan_text.strip().split('\n')
            step_counter = 1

            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Clean up line
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    clean_line = re.sub(r'^[-\*]\s*', '', clean_line)

                    if clean_line:
                        steps.append({
                            'step_number': step_counter,
                            'description': clean_line
                        })
                        step_counter += 1

        return steps

    def create_training_format(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert examples to training format for supervised fine-tuning

        Args:
            examples: Generated planner examples

        Returns:
            List of examples in training format
        """
        training_data = []

        for example in examples:
            # Format for instruction tuning
            instruction = "Generate a strategic plan to solve the following mathematical problem."

            # Create numbered plan text
            plan_text = ""
            for step in example['steps']:
                plan_text += f"{step['step_number']}. {step['description']}\n"

            training_example = {
                'instruction': instruction,
                'input': example['question'],
                'output': plan_text.strip(),
                'metadata': {
                    'source': example['source'],
                    'num_steps': len(example['steps'])
                }
            }

            training_data.append(training_example)

        return training_data

    def augment_with_variations(
        self,
        examples: List[Dict[str, Any]],
        variation_factor: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Create variations of existing examples to increase dataset size

        Args:
            examples: Original examples
            variation_factor: How many variations to create per example

        Returns:
            List including original and variation examples
        """
        augmented_examples = examples.copy()

        for original_example in examples[:len(examples) // 2]:  # Only augment half to avoid overuse
            for _ in range(variation_factor):
                variation_messages = [
                    {
                        "role": "system",
                        "content": """Create a variation of this mathematical problem plan.
Keep the same logical structure but rephrase the steps using different words.
Maintain the same number of steps and overall strategy."""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Plan:
{original_example['plan']}

Create a rephrased version:"""
                    }
                ]

                variation_response = self.call_teacher_model(variation_messages, max_tokens=300)

                if variation_response:
                    variation_steps = self.parse_plan_steps(variation_response)

                    if len(variation_steps) >= 2:
                        variation_example = {
                            'question': original_example['question'],
                            'plan': variation_response,
                            'steps': variation_steps,
                            'original_solution': original_example['original_solution'],
                            'source': 'gsm8k_variation'
                        }

                        augmented_examples.append(variation_example)

        return augmented_examples