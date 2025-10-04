"""
Critic module for ACOR system
Process-supervised evaluation with actionable feedback generation
"""
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .base_module import BaseACORModule
from config.base_config import ACORConfig


class CriticModule(BaseACORModule):
    """
    The Critic module evaluates reasoning steps for correctness and provides
    actionable feedback for error correction and improvement.
    """

    def __init__(
        self,
        config: ACORConfig,
        adapter_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        super().__init__(
            module_name="critic",
            config=config,
            adapter_path=adapter_path,
            device=device
        )

    def create_prompt(
        self,
        question: str,
        step_description: str,
        execution: str,
        context: str = ""
    ) -> str:
        """Create a prompt for step evaluation"""
        system_prompt = """You are a precise critic for mathematical problem solving.
Your task is to evaluate a reasoning step for correctness and provide actionable feedback.

Guidelines:
1. Check if the step execution is logically correct
2. Verify any calculations or mathematical operations
3. Ensure the step addresses what was requested
4. If correct, state "CORRECT" and briefly explain why
5. If incorrect, state "INCORRECT" and provide specific, actionable feedback
6. Focus on what exactly is wrong and how to fix it

Your response format:
STATUS: [CORRECT/INCORRECT]
REASONING: [Your detailed analysis]
FEEDBACK: [Specific guidance for correction if needed]

Example:
Question: "John has 24 cookies and gives 1/3 to his sister. How many does he keep?"
Step: "Calculate how many cookies John gives away"
Execution: "John gives away 1/3 of 24 cookies. 24 ÷ 3 = 6 cookies given away."

STATUS: CORRECT
REASONING: The calculation correctly identifies that 1/3 of 24 is 8, not 6. Wait, let me recalculate: 24 ÷ 3 = 8. The execution shows 24 ÷ 3 = 6, which is incorrect.
FEEDBACK: The division is wrong. 24 ÷ 3 = 8, not 6. Please recalculate 1/3 of 24 cookies.

Now evaluate the following step:"""

        user_prompt = f"""Question: {question}
Step to evaluate: {step_description}
Execution to check: {execution}
Context from previous steps: {context if context else "None"}"""

        return f"{system_prompt}\n\n{user_prompt}"

    def process_input(
        self,
        question: str,
        step_description: str,
        execution: str,
        context: str = ""
    ) -> str:
        """Evaluate a reasoning step and provide feedback"""
        self.logger.info(f"Evaluating step: {step_description[:50]}...")

        prompt = self.create_prompt(question, step_description, execution, context)
        response = self.generate_response(prompt, max_new_tokens=300)

        return response

    def parse_output(self, output: str) -> Dict:
        """Parse the critic output to extract evaluation results"""
        # Extract status
        status_match = re.search(r'STATUS:\s*(CORRECT|INCORRECT)', output, re.IGNORECASE)
        is_correct = False
        if status_match:
            is_correct = status_match.group(1).upper() == "CORRECT"

        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=FEEDBACK:|$)', output, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.+?)$', output, re.DOTALL | re.IGNORECASE)
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        # Fallback parsing if structured format not followed
        if not status_match:
            # Look for correctness indicators
            correct_indicators = ['correct', 'right', 'accurate', 'valid', 'good']
            incorrect_indicators = ['wrong', 'incorrect', 'error', 'mistake', 'invalid', 'bad']

            output_lower = output.lower()
            correct_score = sum(1 for indicator in correct_indicators if indicator in output_lower)
            incorrect_score = sum(1 for indicator in incorrect_indicators if indicator in output_lower)

            is_correct = correct_score > incorrect_score

        # Extract error types if incorrect
        error_types = []
        if not is_correct:
            error_patterns = [
                (r'calculation|arithmetic|math', 'calculation_error'),
                (r'logic|reasoning|step', 'logical_error'),
                (r'misunderstand|misinterpret', 'interpretation_error'),
                (r'missing|incomplete|forgot', 'incomplete_error')
            ]

            for pattern, error_type in error_patterns:
                if re.search(pattern, output.lower()):
                    error_types.append(error_type)

        return {
            'raw_output': output,
            'is_correct': is_correct,
            'reasoning': reasoning,
            'feedback': feedback,
            'error_types': error_types,
            'has_feedback': bool(feedback),
            'confidence': self._assess_confidence(output)
        }

    def _assess_confidence(self, output: str) -> float:
        """Assess the confidence of the critic's evaluation"""
        confidence = 0.5  # Base confidence

        # Higher confidence indicators
        if any(phrase in output.lower() for phrase in [
            'clearly', 'obviously', 'definitely', 'certainly'
        ]):
            confidence += 0.2

        # Lower confidence indicators
        if any(phrase in output.lower() for phrase in [
            'might be', 'possibly', 'unclear', 'hard to tell'
        ]):
            confidence -= 0.2

        # Structured response increases confidence
        if re.search(r'STATUS:\s*(CORRECT|INCORRECT)', output, re.IGNORECASE):
            confidence += 0.1

        # Detailed reasoning increases confidence
        if len(output) > 100:
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def evaluate_step(
        self,
        question: str,
        step_description: str,
        execution: str,
        context: str = ""
    ) -> Dict:
        """
        Main interface method to evaluate a reasoning step

        Args:
            question: The original problem
            step_description: Description of the step being evaluated
            execution: The execution result to evaluate
            context: Context from previous steps

        Returns:
            Dictionary with evaluation results and feedback
        """
        raw_output = self.process_input(question, step_description, execution, context)
        parsed_evaluation = self.parse_output(raw_output)

        # Add metadata
        parsed_evaluation.update({
            'step_description': step_description,
            'execution_length': len(execution),
            'context_available': bool(context)
        })

        self.logger.info(
            f"Step evaluation completed. Correct: {parsed_evaluation['is_correct']}, "
            f"Has feedback: {parsed_evaluation['has_feedback']}"
        )

        return parsed_evaluation

    def get_correction_feedback(self, evaluation_result: Dict) -> str:
        """
        Extract actionable feedback for correction

        Args:
            evaluation_result: Result from evaluate_step

        Returns:
            String with specific feedback for correction
        """
        if evaluation_result['is_correct']:
            return ""

        feedback = evaluation_result.get('feedback', '')
        if not feedback:
            # Generate generic feedback based on error types
            error_types = evaluation_result.get('error_types', [])
            if 'calculation_error' in error_types:
                feedback = "Please double-check your calculations and arithmetic operations."
            elif 'logical_error' in error_types:
                feedback = "Please reconsider the logical approach to this step."
            elif 'interpretation_error' in error_types:
                feedback = "Please re-read the problem and ensure you understand what's being asked."
            elif 'incomplete_error' in error_types:
                feedback = "Please complete all parts of this reasoning step."
            else:
                feedback = "Please review and revise this step for accuracy."

        return feedback

    def batch_evaluate(
        self,
        evaluations: List[Dict]
    ) -> Dict:
        """
        Evaluate multiple steps and provide summary statistics

        Args:
            evaluations: List of evaluation requests with keys:
                - question, step_description, execution, context

        Returns:
            Dictionary with batch evaluation results and statistics
        """
        results = []
        correct_count = 0
        total_feedback_provided = 0

        for eval_request in evaluations:
            result = self.evaluate_step(**eval_request)
            results.append(result)

            if result['is_correct']:
                correct_count += 1
            if result['has_feedback']:
                total_feedback_provided += 1

        total_evaluations = len(evaluations)
        accuracy_rate = correct_count / total_evaluations if total_evaluations > 0 else 0
        feedback_rate = total_feedback_provided / total_evaluations if total_evaluations > 0 else 0

        return {
            'individual_results': results,
            'summary': {
                'total_evaluations': total_evaluations,
                'correct_count': correct_count,
                'accuracy_rate': accuracy_rate,
                'feedback_provided_count': total_feedback_provided,
                'feedback_rate': feedback_rate,
                'error_types_distribution': self._analyze_error_types(results)
            }
        }

    def _analyze_error_types(self, results: List[Dict]) -> Dict:
        """Analyze distribution of error types in evaluation results"""
        error_type_counts = {}

        for result in results:
            for error_type in result.get('error_types', []):
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        return error_type_counts