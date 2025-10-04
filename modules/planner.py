"""
Planner module for ACOR system
Generates high-level, multi-step strategic blueprints for problem solving
"""
import re
from typing import Dict, List, Optional
from pathlib import Path

from .base_module import BaseACORModule
from config.base_config import ACORConfig


class PlannerModule(BaseACORModule):
    """
    The Planner module generates strategic blueprints for solving complex problems.
    It breaks down problems into high-level, sequential steps that guide the execution process.
    """

    def __init__(
        self,
        config: ACORConfig,
        adapter_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        super().__init__(
            module_name="planner",
            config=config,
            adapter_path=adapter_path,
            device=device
        )

    def create_prompt(self, question: str, context: str = "") -> str:
        """Create a prompt for plan generation"""
        system_prompt = """You are a strategic planner for mathematical problem solving.
Your task is to create a high-level, step-by-step plan to solve the given problem.

Guidelines:
1. Break the problem into 3-6 clear, sequential steps
2. Each step should be a high-level action, not detailed calculations
3. Focus on the logical flow and strategy
4. Number each step clearly
5. Keep steps concise but descriptive

Example:
Problem: "Sarah has 3 boxes with 12 apples each. She gives away 8 apples. How many apples does she have left?"
Plan:
1. Calculate total apples in all boxes
2. Subtract the apples given away
3. Determine the final count

Now create a plan for the following problem:"""

        user_prompt = f"Problem: {question}"
        if context:
            user_prompt += f"\nAdditional Context: {context}"

        return f"{system_prompt}\n\n{user_prompt}\n\nPlan:"

    def process_input(self, question: str, context: str = "") -> str:
        """Generate a strategic plan for the given question"""
        self.logger.info(f"Generating plan for question: {question[:100]}...")

        prompt = self.create_prompt(question, context)
        response = self.generate_response(prompt, max_new_tokens=256)

        return response

    def parse_output(self, output: str) -> Dict:
        """Parse the plan output into structured steps"""
        # Extract numbered steps from the output
        steps = []

        # Look for numbered patterns like "1.", "2)", "Step 1:", etc.
        step_patterns = [
            r'(\d+)\.\s*(.+?)(?=\n\d+\.|\n[A-Z]|\Z)',  # "1. step text"
            r'(\d+)\)\s*(.+?)(?=\n\d+\)|\n[A-Z]|\Z)',  # "1) step text"
            r'Step\s+(\d+):?\s*(.+?)(?=\nStep|\n[A-Z]|\Z)',  # "Step 1: text"
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                for match in matches:
                    step_num = int(match[0])
                    step_text = match[1].strip().replace('\n', ' ')
                    steps.append({
                        'step_number': step_num,
                        'description': step_text
                    })
                break

        # Fallback: split by lines and look for step-like content
        if not steps:
            lines = output.strip().split('\n')
            step_counter = 1
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Plan:'):
                    # Remove any leading numbers/bullets
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    clean_line = re.sub(r'^[-\*]\s*', '', clean_line)

                    if clean_line:
                        steps.append({
                            'step_number': step_counter,
                            'description': clean_line
                        })
                        step_counter += 1

        return {
            'raw_output': output,
            'steps': steps,
            'num_steps': len(steps)
        }

    def generate_plan(self, question: str, context: str = "") -> List[Dict]:
        """
        Main interface method to generate a plan for a given question

        Args:
            question: The problem to solve
            context: Additional context or constraints

        Returns:
            List of plan steps with structure and descriptions
        """
        raw_output = self.process_input(question, context)
        parsed_plan = self.parse_output(raw_output)

        self.logger.info(f"Generated plan with {parsed_plan['num_steps']} steps")

        return parsed_plan['steps']

    def validate_plan(self, steps: List[Dict], question: str) -> Dict:
        """
        Validate if the generated plan is reasonable for the given question

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }

        # Check if we have steps
        if not steps:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No steps generated")
            return validation_results

        # Check for reasonable number of steps (2-8 is typical)
        if len(steps) < 2:
            validation_results['issues'].append("Plan may be too simple (< 2 steps)")
        elif len(steps) > 8:
            validation_results['issues'].append("Plan may be too complex (> 8 steps)")

        # Check if steps are sequentially numbered
        expected_nums = list(range(1, len(steps) + 1))
        actual_nums = [step['step_number'] for step in steps]
        if actual_nums != expected_nums:
            validation_results['issues'].append("Steps are not sequentially numbered")

        # Check for very short or empty descriptions
        for step in steps:
            if len(step['description']) < 10:
                validation_results['issues'].append(
                    f"Step {step['step_number']} description is too brief"
                )

        # Set validity based on critical issues
        critical_issues = [issue for issue in validation_results['issues']
                          if any(critical in issue.lower()
                                for critical in ['no steps', 'empty'])]
        validation_results['is_valid'] = len(critical_issues) == 0

        return validation_results