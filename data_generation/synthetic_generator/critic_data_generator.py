"""
Generate training data for the Critic module using synthetic error injection
"""
import json
import re
import random
from typing import Dict, List, Any, Tuple
from pathlib import Path

from ..base_generator import BaseDataGenerator


class CriticDataGenerator(BaseDataGenerator):
    """Generate (flawed_reasoning_step, actionable_feedback) pairs for Critic module training"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define error types and patterns
        self.error_types = [
            'calculation_error',
            'logical_error',
            'interpretation_error',
            'incomplete_step',
            'wrong_operation',
            'unit_error'
        ]

    def _validate_example_impl(self, example: Dict[str, Any]) -> bool:
        """Validate a critic training example"""
        required_fields = ['question', 'step_description', 'correct_execution',
                          'flawed_execution', 'feedback', 'error_type']

        # Check required fields
        for field in required_fields:
            if field not in example:
                return False

        # Check content quality
        flawed_execution = example['flawed_execution']
        feedback = example['feedback']

        # Execution validation
        if not flawed_execution or len(flawed_execution) < 30:
            return False

        # Feedback validation
        if not feedback or len(feedback) < 20:
            return False

        # Check that feedback is actionable
        actionable_indicators = [
            'should', 'need to', 'must', 'recalculate', 'check', 'correct',
            'instead', 'rather', 'fix', 'revise', 'change'
        ]

        if not any(indicator in feedback.lower() for indicator in actionable_indicators):
            return False

        return True

    def create_correct_solution_prompt(self, question: str) -> List[Dict[str, str]]:
        """Create prompt for generating correct step-by-step solution"""
        return [
            {
                "role": "system",
                "content": """Solve this mathematical problem step-by-step with clear reasoning.
Show all calculations and explain each step thoroughly.
Be precise with numbers and operations."""
            },
            {
                "role": "user",
                "content": f"Solve this problem step-by-step:\n{question}"
            }
        ]

    def create_error_injection_prompt(
        self,
        question: str,
        correct_solution: str,
        error_type: str
    ) -> List[Dict[str, str]]:
        """Create prompt for injecting specific type of error"""

        error_instructions = {
            'calculation_error': "Introduce a calculation mistake (wrong arithmetic, incorrect number)",
            'logical_error': "Introduce a logical flaw in the reasoning approach",
            'interpretation_error': "Misinterpret part of the problem statement",
            'incomplete_step': "Leave out important details or skip part of the calculation",
            'wrong_operation': "Use the wrong mathematical operation (+ instead of ×, etc.)",
            'unit_error': "Make an error with units or conversions"
        }

        instruction = error_instructions.get(error_type, "Introduce a plausible mathematical error")

        return [
            {
                "role": "system",
                "content": f"""Take this correct solution and introduce a specific error.
The error should be: {instruction}

Make the error subtle and plausible - something a student might realistically do.
Keep the overall structure and format similar to the original.
Only introduce ONE clear error."""
            },
            {
                "role": "user",
                "content": f"""Problem: {question}
Correct solution: {correct_solution}

Create a version with the specified error type ({error_type}):"""
            }
        ]

    def create_feedback_generation_prompt(
        self,
        question: str,
        step_description: str,
        correct_execution: str,
        flawed_execution: str
    ) -> List[Dict[str, str]]:
        """Create prompt for generating actionable feedback"""
        return [
            {
                "role": "system",
                "content": """You are reviewing a student's work on a mathematical problem.
Compare the correct approach with the flawed attempt and provide specific, actionable feedback.

Your feedback should:
1. Identify what specific error was made
2. Explain why it's incorrect
3. Provide clear guidance on how to fix it
4. Be constructive and educational

Example format:
"The calculation is incorrect. You computed 24 ÷ 3 = 6, but 24 ÷ 3 = 8. Please recalculate the division to find the correct answer."
"""
            },
            {
                "role": "user",
                "content": f"""Problem: {question}
Step being executed: {step_description}

CORRECT execution:
{correct_execution}

STUDENT'S flawed execution:
{flawed_execution}

Provide specific, actionable feedback to help the student correct their error:"""
            }
        ]

    def generate_examples(
        self,
        input_data: List[Dict[str, Any]],
        num_examples: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate critic training examples using error injection

        Args:
            input_data: List of GSM8K problems or math problems
            num_examples: Number of examples to generate

        Returns:
            List of critic training examples
        """
        examples = []
        examples_per_problem = kwargs.get('examples_per_problem', 3)

        for problem_data in input_data:
            if len(examples) >= num_examples:
                break

            question = problem_data.get('question', '')
            if not question:
                continue

            # Generate correct solution
            correct_messages = self.create_correct_solution_prompt(question)
            correct_solution = self.call_teacher_model(correct_messages, max_tokens=500)

            if not correct_solution:
                continue

            # Break solution into steps
            solution_steps = self.extract_solution_steps(correct_solution)

            # Generate error examples for each step
            for step_idx, step_content in enumerate(solution_steps[:examples_per_problem]):
                if len(examples) >= num_examples:
                    break

                # Randomly select error type
                error_type = random.choice(self.error_types)

                # Create flawed version
                error_messages = self.create_error_injection_prompt(
                    question, step_content, error_type
                )
                flawed_execution = self.call_teacher_model(error_messages, max_tokens=400)

                if not flawed_execution:
                    continue

                # Generate feedback
                step_description = f"Step {step_idx + 1} of the solution"
                feedback_messages = self.create_feedback_generation_prompt(
                    question, step_description, step_content, flawed_execution
                )
                feedback = self.call_teacher_model(feedback_messages, max_tokens=300)

                if not feedback:
                    continue

                # Create training example
                training_example = {
                    'question': question,
                    'step_description': step_description,
                    'correct_execution': step_content,
                    'flawed_execution': flawed_execution,
                    'feedback': feedback,
                    'error_type': error_type,
                    'source': 'synthetic_error_injection'
                }

                examples.append(training_example)

            if len(examples) % 25 == 0:
                self.logger.info(f"Generated {len(examples)} critic examples")

        return examples

    def extract_solution_steps(self, solution: str) -> List[str]:
        """Extract individual steps from a complete solution"""
        # Split by sentences and logical breaks
        sentences = re.split(r'[.!?]+', solution)

        steps = []
        current_step = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if this starts a new calculation step
            has_calculation = bool(re.search(r'\d+.*[+\-*/×÷=].*\d+', sentence))
            is_conclusion = any(word in sentence.lower() for word in
                              ['therefore', 'so', 'thus', 'finally', 'result'])

            if (has_calculation or is_conclusion) and current_step:
                # Save current step and start new one
                if len(current_step.strip()) > 20:
                    steps.append(current_step.strip())
                current_step = sentence
            else:
                # Continue building current step
                if current_step:
                    current_step += " " + sentence
                else:
                    current_step = sentence

        # Add final step
        if current_step and len(current_step.strip()) > 20:
            steps.append(current_step.strip())

        return steps

    def create_training_format(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert examples to training format for supervised fine-tuning

        Args:
            examples: Generated critic examples

        Returns:
            List of examples in training format
        """
        training_data = []

        for example in examples:
            # Create evaluation prompt
            instruction = """Evaluate the following reasoning step for correctness and provide feedback.
Respond with:
STATUS: [CORRECT/INCORRECT]
REASONING: [Your analysis]
FEEDBACK: [Specific guidance if incorrect]"""

            # Input contains question, step, and execution to evaluate
            input_text = f"""Question: {example['question']}
Step: {example['step_description']}
Execution: {example['flawed_execution']}"""

            # Output is structured evaluation with feedback
            output_text = f"""STATUS: INCORRECT
REASONING: {self._extract_reasoning_from_feedback(example['feedback'])}
FEEDBACK: {example['feedback']}"""

            training_example = {
                'instruction': instruction,
                'input': input_text,
                'output': output_text,
                'metadata': {
                    'error_type': example['error_type'],
                    'source': example['source']
                }
            }

            training_data.append(training_example)

            # Also create positive examples (correct executions)
            if random.random() < 0.3:  # 30% positive examples
                positive_output = f"""STATUS: CORRECT
REASONING: The execution correctly follows the required approach and shows accurate calculations.
FEEDBACK: The step is well-executed and mathematically sound."""

                positive_example = {
                    'instruction': instruction,
                    'input': input_text.replace(example['flawed_execution'], example['correct_execution']),
                    'output': positive_output,
                    'metadata': {
                        'error_type': 'none',
                        'source': 'positive_example'
                    }
                }

                training_data.append(positive_example)

        return training_data

    def _extract_reasoning_from_feedback(self, feedback: str) -> str:
        """Extract reasoning explanation from feedback"""
        # Try to identify the explanation part of the feedback
        if '.' in feedback:
            first_sentence = feedback.split('.')[0]
            return first_sentence + "."
        return "The approach contains an error that needs to be addressed."

    def generate_diverse_error_examples(
        self,
        base_examples: List[Dict[str, Any]],
        diversity_factor: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate additional examples with diverse error patterns

        Args:
            base_examples: Original examples
            diversity_factor: Multiplier for additional examples

        Returns:
            Extended list with more diverse error examples
        """
        diverse_examples = base_examples.copy()

        # Create variations of existing errors
        for original in base_examples[:len(base_examples) // 2]:
            for _ in range(diversity_factor):
                # Create variation with different error type
                available_errors = [et for et in self.error_types if et != original['error_type']]
                new_error_type = random.choice(available_errors)

                variation_messages = self.create_error_injection_prompt(
                    original['question'],
                    original['correct_execution'],
                    new_error_type
                )

                variation_execution = self.call_teacher_model(variation_messages, max_tokens=400)

                if variation_execution:
                    # Generate new feedback
                    feedback_messages = self.create_feedback_generation_prompt(
                        original['question'],
                        original['step_description'],
                        original['correct_execution'],
                        variation_execution
                    )

                    variation_feedback = self.call_teacher_model(feedback_messages, max_tokens=300)

                    if variation_feedback:
                        diverse_example = {
                            'question': original['question'],
                            'step_description': original['step_description'],
                            'correct_execution': original['correct_execution'],
                            'flawed_execution': variation_execution,
                            'feedback': variation_feedback,
                            'error_type': new_error_type,
                            'source': 'error_variation'
                        }

                        diverse_examples.append(diverse_example)

        return diverse_examples