"""
Generate training data for the Executor module from GSM8K
"""
import json
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

from ..base_generator import BaseDataGenerator


class ExecutorDataGenerator(BaseDataGenerator):
    """Generate (question + plan_step, execution) pairs for Executor module training"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _validate_example_impl(self, example: Dict[str, Any]) -> bool:
        """Validate an executor training example"""
        required_fields = ['question', 'step_description', 'execution', 'context']

        # Check required fields
        for field in required_fields:
            if field not in example:
                return False

        # Check content quality
        question = example['question']
        step_description = example['step_description']
        execution = example['execution']

        # Question validation
        if not question or len(question) < 20:
            return False

        # Step description validation
        if not step_description or len(step_description) < 10:
            return False

        # Execution validation
        if not execution or len(execution) < 30:
            return False

        # Check for mathematical content in execution for math problems
        if any(word in step_description.lower() for word in ['calculate', 'find', 'compute']):
            if not re.search(r'\d+', execution):
                return False

        return True

    def create_execution_generation_prompt(
        self,
        question: str,
        step_description: str,
        context: str,
        reference_solution: str
    ) -> List[Dict[str, str]]:
        """Create prompt for generating step execution"""
        return [
            {
                "role": "system",
                "content": """You are a precise executor for mathematical problem solving.
Your task is to execute a specific step with detailed work and clear reasoning.

Guidelines:
1. Focus ONLY on the current step - don't solve the entire problem
2. Show all calculations and mathematical operations clearly
3. Use information from context if needed
4. Be precise with numbers and arithmetic
5. State your result clearly at the end
6. Show your work step-by-step

Example:
Question: "Sarah has 3 boxes with 12 apples each. She gives away 8 apples. How many apples left?"
Step: "Calculate total apples in all boxes"
Context: ""

Execution:
I need to find the total number of apples across all 3 boxes.
- Number of boxes: 3
- Apples per box: 12
- Total apples = 3 × 12 = 36 apples

Result: Sarah initially has 36 apples total."""
            },
            {
                "role": "user",
                "content": f"""Question: {question}
Step to execute: {step_description}
Context from previous steps: {context if context else "None"}

Based on the reference solution, execute this specific step with detailed work:
Reference (for guidance): {reference_solution}

Execution:"""
            }
        ]

    def extract_step_executions_from_solution(
        self,
        question: str,
        plan_steps: List[Dict[str, Any]],
        solution: str
    ) -> List[Dict[str, Any]]:
        """
        Extract step-by-step executions from complete solution

        Args:
            question: The problem question
            plan_steps: List of plan steps
            solution: Complete solution text

        Returns:
            List of step execution examples
        """
        executions = []

        # Split solution into sentences for analysis
        sentences = re.split(r'[.!?]+', solution)
        solution_parts = [s.strip() for s in sentences if s.strip()]

        context = ""
        used_parts = set()

        for step_idx, step in enumerate(plan_steps):
            step_description = step['description']

            # Generate execution for this step
            messages = self.create_execution_generation_prompt(
                question, step_description, context, solution
            )

            execution = self.call_teacher_model(messages, max_tokens=400)

            if execution:
                execution_example = {
                    'question': question,
                    'step_number': step['step_number'],
                    'step_description': step_description,
                    'context': context,
                    'execution': execution,
                    'reference_solution': solution
                }

                executions.append(execution_example)

                # Update context with this execution for next step
                context += f"Step {step['step_number']}: {execution}\n\n"

        return executions

    def generate_examples(
        self,
        input_data: List[Dict[str, Any]],
        num_examples: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate executor training examples

        Args:
            input_data: List of GSM8K examples with extracted plans
            num_examples: Target number of examples

        Returns:
            List of executor training examples
        """
        examples = []
        examples_per_problem = kwargs.get('examples_per_problem', 4)

        for gsm8k_example in input_data:
            if len(examples) >= num_examples:
                break

            question = gsm8k_example['question']
            solution = gsm8k_example['solution']

            # Generate plan for this problem first
            plan_messages = self.create_plan_extraction_prompt(question, solution)
            plan_response = self.call_teacher_model(plan_messages, max_tokens=300)

            if not plan_response:
                continue

            plan_steps = self.parse_plan_steps(plan_response)
            if len(plan_steps) < 2:
                continue

            # Generate executions for each step
            step_executions = self.extract_step_executions_from_solution(
                question, plan_steps, solution
            )

            # Add valid executions to examples
            for execution_example in step_executions:
                if self._validate_example_impl(execution_example):
                    examples.append(execution_example)

                if len(examples) >= num_examples:
                    break

            if len(examples) % 50 == 0:
                self.logger.info(f"Generated {len(examples)} executor examples")

        return examples

    def create_plan_extraction_prompt(self, question: str, solution: str) -> List[Dict[str, str]]:
        """Create prompt for extracting plan (reused from planner)"""
        return [
            {
                "role": "system",
                "content": """Extract a high-level plan from this mathematical solution.
Create 3-6 clear, sequential steps that describe the problem-solving strategy."""
            },
            {
                "role": "user",
                "content": f"""Problem: {question}
Solution: {solution}

Extract a strategic plan:"""
            }
        ]

    def parse_plan_steps(self, plan_text: str) -> List[Dict[str, Any]]:
        """Parse plan text into structured steps (reused from planner)"""
        import re

        steps = []
        step_patterns = [
            r'(\d+)\.\s*(.+?)(?=\n\d+\.|\n[A-Z]|\Z)',
            r'(\d+)\)\s*(.+?)(?=\n\d+\)|\n[A-Z]|\Z)',
            r'Step\s+(\d+):?\s*(.+?)(?=\nStep|\n[A-Z]|\Z)',
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, plan_text, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    step_num = int(match[0])
                    step_text = match[1].strip().replace('\n', ' ')
                    if len(step_text) > 5:
                        steps.append({
                            'step_number': step_num,
                            'description': step_text
                        })
                break

        return steps

    def create_training_format(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert examples to training format for supervised fine-tuning

        Args:
            examples: Generated executor examples

        Returns:
            List of examples in training format
        """
        training_data = []

        for example in examples:
            # Format instruction with context
            instruction = "Execute the following reasoning step with detailed work."

            # Create input with question, step, and context
            input_text = f"""Question: {example['question']}
Step to execute: {example['step_description']}
Context: {example['context'] if example['context'] else "None"}"""

            training_example = {
                'instruction': instruction,
                'input': input_text,
                'output': example['execution'],
                'metadata': {
                    'step_number': example['step_number'],
                    'has_context': bool(example['context'])
                }
            }

            training_data.append(training_example)

        return training_data

    def generate_context_variations(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate examples with different context lengths to improve robustness

        Args:
            examples: Original executor examples

        Returns:
            Augmented examples with context variations
        """
        augmented_examples = examples.copy()

        # Create examples with truncated context
        for example in examples[:len(examples) // 3]:
            if example['context']:
                # Create version with shorter context
                context_parts = example['context'].split('\n\n')
                if len(context_parts) > 1:
                    short_context = context_parts[-1]  # Only most recent step

                    variation_example = example.copy()
                    variation_example['context'] = short_context
                    variation_example['source'] = 'context_variation'

                    augmented_examples.append(variation_example)

        return augmented_examples

    def create_error_correction_examples(
        self,
        examples: List[Dict[str, Any]],
        error_rate: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Create examples with intentional errors for training robustness

        Args:
            examples: Original examples
            error_rate: Fraction of examples to create error versions for

        Returns:
            Examples including some with intentional errors for learning
        """
        error_examples = []
        num_error_examples = int(len(examples) * error_rate)

        for i in range(num_error_examples):
            original = examples[i % len(examples)]

            error_messages = [
                {
                    "role": "system",
                    "content": """Create a version of this execution with a subtle mathematical error.
The error should be plausible but incorrect (wrong operation, calculation mistake, etc.).
Keep the format and structure the same."""
                },
                {
                    "role": "user",
                    "content": f"""Original correct execution:
{original['execution']}

Create a version with a subtle error:"""
                }
            ]

            error_execution = self.call_teacher_model(error_messages, max_tokens=400)

            if error_execution:
                error_example = original.copy()
                error_example['execution'] = error_execution
                error_example['is_error'] = True
                error_example['correct_execution'] = original['execution']

                error_examples.append(error_example)

        return error_examples