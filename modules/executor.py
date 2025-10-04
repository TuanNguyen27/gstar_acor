"""
Executor module for ACOR system
Executes single, well-defined reasoning steps based on plan and context
"""
import re
from typing import Dict, List, Optional
from pathlib import Path

from .base_module import BaseACORModule
from config.base_config import ACORConfig


class ExecutorModule(BaseACORModule):
    """
    The Executor module performs detailed execution of individual reasoning steps.
    It takes a plan step and current context to produce specific calculations and reasoning.
    """

    def __init__(
        self,
        config: ACORConfig,
        adapter_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        super().__init__(
            module_name="executor",
            config=config,
            adapter_path=adapter_path,
            device=device
        )

    def create_prompt(
        self,
        question: str,
        step_description: str,
        context: str = "",
        previous_feedback: str = ""
    ) -> str:
        """Create a prompt for step execution"""
        system_prompt = """You are a precise executor for mathematical problem solving.
Your task is to execute a specific step from a problem-solving plan with detailed work.

Guidelines:
1. Focus ONLY on the current step - don't solve the entire problem
2. Show all calculations and reasoning clearly
3. Be precise with numbers and operations
4. If the step involves calculations, show the arithmetic
5. State your result clearly at the end
6. If you need information from previous steps, use the provided context

Example:
Question: "Sarah has 3 boxes with 12 apples each. She gives away 8 apples. How many apples left?"
Step: "Calculate total apples in all boxes"
Context: ""

Execution:
I need to find the total number of apples across all boxes.
- Number of boxes: 3
- Apples per box: 12
- Total apples = 3 × 12 = 36 apples

Result: Sarah initially has 36 apples total.

Now execute the following step:"""

        user_prompt = f"""Question: {question}
Step to execute: {step_description}
Context from previous steps: {context if context else "None"}"""

        if previous_feedback:
            user_prompt += f"\nPrevious feedback to address: {previous_feedback}"

        user_prompt += "\n\nExecution:"

        return f"{system_prompt}\n\n{user_prompt}"

    def process_input(
        self,
        question: str,
        step_description: str,
        context: str = "",
        previous_feedback: str = ""
    ) -> str:
        """Execute a specific step with detailed reasoning"""
        self.logger.info(f"Executing step: {step_description[:50]}...")

        prompt = self.create_prompt(question, step_description, context, previous_feedback)
        response = self.generate_response(prompt, max_new_tokens=400)

        return response

    def parse_output(self, output: str) -> Dict:
        """Parse the execution output to extract key information"""
        # Look for calculations and results
        calculations = []
        result = ""

        # Extract mathematical expressions and calculations
        calc_patterns = [
            r'(\d+(?:\.\d+)?)\s*[×x*]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[+]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[-−]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*[÷/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        ]

        for pattern in calc_patterns:
            matches = re.findall(pattern, output)
            calculations.extend(matches)

        # Extract the final result
        result_patterns = [
            r'Result:?\s*(.+?)(?:\n|$)',
            r'Answer:?\s*(.+?)(?:\n|$)',
            r'Therefore,?\s*(.+?)(?:\n|$)',
            r'Final(?:ly)?:?\s*(.+?)(?:\n|$)'
        ]

        for pattern in result_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                break

        # Extract any numbers mentioned (for numerical problems)
        numbers = re.findall(r'\d+(?:\.\d+)?', output)

        # Look for key mathematical operations
        operations = {
            'multiplication': bool(re.search(r'[×x*]', output)),
            'addition': bool(re.search(r'[+]', output)),
            'subtraction': bool(re.search(r'[-−]', output)),
            'division': bool(re.search(r'[÷/]', output))
        }

        return {
            'raw_output': output,
            'calculations': calculations,
            'result': result,
            'numbers_used': numbers,
            'operations': operations,
            'has_result': bool(result)
        }

    def execute_step(
        self,
        question: str,
        step_description: str,
        context: str = "",
        previous_feedback: str = ""
    ) -> Dict:
        """
        Main interface method to execute a reasoning step

        Args:
            question: The original problem
            step_description: Description of the step to execute
            context: Context from previous steps
            previous_feedback: Feedback from critic for revision

        Returns:
            Dictionary with execution results and analysis
        """
        raw_output = self.process_input(question, step_description, context, previous_feedback)
        parsed_execution = self.parse_output(raw_output)

        # Add metadata
        parsed_execution.update({
            'step_description': step_description,
            'used_feedback': bool(previous_feedback),
            'context_length': len(context) if context else 0
        })

        self.logger.info(f"Step execution completed. Has result: {parsed_execution['has_result']}")

        return parsed_execution

    def validate_execution(
        self,
        execution_result: Dict,
        step_description: str
    ) -> Dict:
        """
        Validate the quality of step execution

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'confidence': 0.0,
            'issues': [],
            'strengths': []
        }

        # Check if result was produced
        if not execution_result['has_result']:
            validation_results['issues'].append("No clear result stated")
            validation_results['confidence'] -= 0.3

        # Check for mathematical content in math problems
        if any(word in step_description.lower() for word in ['calculate', 'find', 'compute', 'total']):
            if not execution_result['calculations'] and not execution_result['numbers_used']:
                validation_results['issues'].append("Mathematical step lacks numerical work")
                validation_results['confidence'] -= 0.4
            else:
                validation_results['strengths'].append("Shows mathematical calculations")
                validation_results['confidence'] += 0.3

        # Check output length - too short might indicate incomplete work
        output_length = len(execution_result['raw_output'])
        if output_length < 50:
            validation_results['issues'].append("Execution appears too brief")
            validation_results['confidence'] -= 0.2
        elif output_length > 20:
            validation_results['strengths'].append("Provides detailed explanation")
            validation_results['confidence'] += 0.1

        # Check for logical flow indicators
        logical_indicators = ['because', 'therefore', 'since', 'so', 'thus', 'then']
        if any(indicator in execution_result['raw_output'].lower() for indicator in logical_indicators):
            validation_results['strengths'].append("Shows logical reasoning")
            validation_results['confidence'] += 0.2

        # Final confidence calculation
        base_confidence = 0.7
        validation_results['confidence'] = max(0.0, min(1.0, base_confidence + validation_results['confidence']))

        # Determine validity
        critical_issues = [issue for issue in validation_results['issues']
                          if any(critical in issue.lower()
                                for critical in ['no clear result', 'lacks numerical'])]
        validation_results['is_valid'] = len(critical_issues) == 0 and validation_results['confidence'] > 0.3

        return validation_results