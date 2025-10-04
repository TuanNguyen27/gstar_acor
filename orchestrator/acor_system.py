"""
Main ACOR system orchestrator
Coordinates Planner, Executor, and Critic modules in iterative refinement loop
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..modules.planner import PlannerModule
from ..modules.executor import ExecutorModule
from ..modules.critic import CriticModule
from config.base_config import ACORConfig


class ACORSystem:
    """
    Main ACOR system that orchestrates the Planner-Executor-Critic loop
    with dynamic iterative refinement for self-correction
    """

    def __init__(
        self,
        planner: PlannerModule,
        executor: ExecutorModule,
        critic: CriticModule,
        config: ACORConfig
    ):
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.config = config

        # Set up logging
        self.logger = logging.getLogger("acor.orchestrator")

        # Statistics tracking
        self.stats = {
            'problems_solved': 0,
            'total_steps': 0,
            'total_corrections': 0,
            'successful_corrections': 0,
            'total_execution_time': 0.0
        }

    def solve_problem(
        self,
        question: str,
        enable_correction: bool = True,
        max_correction_attempts: int = 3,
        timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """
        Solve a problem using the ACOR system

        Args:
            question: Problem to solve
            enable_correction: Whether to use self-correction loop
            max_correction_attempts: Maximum correction attempts per step
            timeout_seconds: Maximum time to spend on problem

        Returns:
            Dictionary with solution results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Solving problem: {question[:100]}...")

        try:
            # Initialize result structure
            result = {
                'question': question,
                'success': False,
                'steps': [],
                'final_answer': None,
                'execution_time': 0.0,
                'total_attempts': 0,
                'corrections_made': 0,
                'error_message': None
            }

            # Step 1: Generate plan
            self.logger.info("Generating plan...")
            plan_steps = self.planner.generate_plan(question)

            if not plan_steps:
                result['error_message'] = "Failed to generate valid plan"
                return result

            self.logger.info(f"Generated plan with {len(plan_steps)} steps")

            # Step 2: Execute plan with iterative refinement
            context = ""
            all_executions = []

            for step_idx, plan_step in enumerate(plan_steps):
                step_description = plan_step['description']
                self.logger.info(f"Executing step {step_idx + 1}: {step_description[:50]}...")

                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    result['error_message'] = f"Timeout after {timeout_seconds}s"
                    break

                # Execute step with correction loop
                step_result = self._execute_step_with_correction(
                    question=question,
                    step_description=step_description,
                    context=context,
                    enable_correction=enable_correction,
                    max_attempts=max_correction_attempts
                )

                result['steps'].append(step_result)
                result['total_attempts'] += step_result['total_attempts']
                result['corrections_made'] += step_result['corrections_made']

                # Update context with successful execution
                if step_result['final_execution']:
                    context += f"Step {step_idx + 1}: {step_result['final_execution']}\n\n"
                    all_executions.append(step_result['final_execution'])

            # Step 3: Extract final answer
            final_answer = self._extract_final_answer(all_executions, question)
            result['final_answer'] = final_answer
            result['success'] = bool(final_answer and not result.get('error_message'))

            # Update statistics
            self.stats['problems_solved'] += 1
            self.stats['total_steps'] += len(result['steps'])
            self.stats['total_corrections'] += result['corrections_made']

        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
            result['error_message'] = str(e)
            result['success'] = False

        finally:
            result['execution_time'] = time.time() - start_time
            self.stats['total_execution_time'] += result['execution_time']

        self.logger.info(
            f"Problem completed. Success: {result['success']}, "
            f"Time: {result['execution_time']:.2f}s, "
            f"Corrections: {result['corrections_made']}"
        )

        return result

    def _execute_step_with_correction(
        self,
        question: str,
        step_description: str,
        context: str,
        enable_correction: bool,
        max_attempts: int
    ) -> Dict[str, Any]:
        """
        Execute a single step with correction loop

        Args:
            question: Original problem question
            step_description: Description of step to execute
            context: Context from previous steps
            enable_correction: Whether to use correction loop
            max_attempts: Maximum correction attempts

        Returns:
            Dictionary with step execution results
        """
        step_result = {
            'step_description': step_description,
            'attempts': [],
            'final_execution': None,
            'total_attempts': 0,
            'corrections_made': 0,
            'success': False
        }

        previous_feedback = ""

        for attempt in range(max_attempts):
            attempt_num = attempt + 1
            step_result['total_attempts'] = attempt_num

            self.logger.debug(f"Step attempt {attempt_num}/{max_attempts}")

            # Execute the step
            execution_result = self.executor.execute_step(
                question=question,
                step_description=step_description,
                context=context,
                previous_feedback=previous_feedback
            )

            execution_text = execution_result['raw_output']

            # Create attempt record
            attempt_record = {
                'attempt_number': attempt_num,
                'execution': execution_text,
                'execution_metadata': execution_result,
                'critic_evaluation': None,
                'feedback': None,
                'is_correct': False
            }

            # Evaluate with critic if correction is enabled
            if enable_correction:
                critic_result = self.critic.evaluate_step(
                    question=question,
                    step_description=step_description,
                    execution=execution_text,
                    context=context
                )

                attempt_record['critic_evaluation'] = critic_result
                attempt_record['is_correct'] = critic_result['is_correct']

                # If incorrect and not the last attempt, get feedback for correction
                if not critic_result['is_correct'] and attempt < max_attempts - 1:
                    feedback = self.critic.get_correction_feedback(critic_result)
                    attempt_record['feedback'] = feedback
                    previous_feedback = feedback
                    step_result['corrections_made'] += 1

                    self.logger.debug(f"Correction needed. Feedback: {feedback[:100]}...")
                else:
                    # Either correct or last attempt
                    step_result['final_execution'] = execution_text
                    step_result['success'] = critic_result['is_correct']
                    break

            else:
                # No correction - accept first attempt
                attempt_record['is_correct'] = True  # Assume correct without critic
                step_result['final_execution'] = execution_text
                step_result['success'] = True

            step_result['attempts'].append(attempt_record)

            # If correction disabled, only do one attempt
            if not enable_correction:
                break

        # If no successful execution, use the last attempt
        if not step_result['final_execution'] and step_result['attempts']:
            step_result['final_execution'] = step_result['attempts'][-1]['execution']

        return step_result

    def _extract_final_answer(
        self,
        executions: List[str],
        question: str
    ) -> Optional[str]:
        """
        Extract final answer from all step executions

        Args:
            executions: List of step executions
            question: Original question

        Returns:
            Final answer string or None if not found
        """
        if not executions:
            return None

        # Combine all executions
        full_solution = "\n".join(executions)

        # Look for explicit final answers
        import re

        # Common final answer patterns
        patterns = [
            r'(?:final (?:answer|result)|answer|result|therefore|thus|so).*?(\d+(?:\.\d+)?)',
            r'=\s*(\d+(?:\.\d+)?)\s*(?:$|\.|,)',
            r'(\d+(?:\.\d+)?)\s*(?:dollars?|cents?|items?|people?|things?)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, full_solution, re.IGNORECASE)
            if matches:
                return matches[-1]  # Return the last match

        # If no explicit answer found, look for the last number mentioned
        numbers = re.findall(r'\d+(?:\.\d+)?', full_solution)
        if numbers:
            return numbers[-1]

        return "Solution completed (see steps above)"

    def solve_batch(
        self,
        problems: List[str],
        enable_correction: bool = True,
        max_problems: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Solve a batch of problems

        Args:
            problems: List of problem questions
            enable_correction: Whether to use self-correction
            max_problems: Maximum number of problems to solve
            progress_callback: Optional callback for progress updates

        Returns:
            List of solution results
        """
        if max_problems:
            problems = problems[:max_problems]

        self.logger.info(f"Solving batch of {len(problems)} problems")

        results = []

        for i, problem in enumerate(problems):
            if progress_callback:
                progress_callback(i, len(problems))

            result = self.solve_problem(problem, enable_correction=enable_correction)
            results.append(result)

            # Log progress
            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r['success']) / len(results)
                avg_time = sum(r['execution_time'] for r in results) / len(results)
                self.logger.info(
                    f"Progress: {i + 1}/{len(problems)}, "
                    f"Success rate: {success_rate:.1%}, "
                    f"Avg time: {avg_time:.1f}s"
                )

        final_success_rate = sum(1 for r in results if r['success']) / len(results)
        self.logger.info(f"Batch completed. Final success rate: {final_success_rate:.1%}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.stats.copy()

        # Calculate derived statistics
        if stats['problems_solved'] > 0:
            stats['avg_steps_per_problem'] = stats['total_steps'] / stats['problems_solved']
            stats['avg_corrections_per_problem'] = stats['total_corrections'] / stats['problems_solved']
            stats['avg_time_per_problem'] = stats['total_execution_time'] / stats['problems_solved']
        else:
            stats['avg_steps_per_problem'] = 0
            stats['avg_corrections_per_problem'] = 0
            stats['avg_time_per_problem'] = 0

        if stats['total_corrections'] > 0:
            stats['correction_success_rate'] = stats['successful_corrections'] / stats['total_corrections']
        else:
            stats['correction_success_rate'] = 0

        return stats

    def reset_statistics(self):
        """Reset system statistics"""
        self.stats = {
            'problems_solved': 0,
            'total_steps': 0,
            'total_corrections': 0,
            'successful_corrections': 0,
            'total_execution_time': 0.0
        }

    def validate_system_health(self) -> Dict[str, bool]:
        """
        Validate that all system components are working

        Returns:
            Dictionary with health status for each component
        """
        health = {
            'planner_healthy': False,
            'executor_healthy': False,
            'critic_healthy': False,
            'system_healthy': False
        }

        try:
            # Test planner
            test_question = "If John has 5 apples and buys 3 more, how many apples does he have?"
            plan = self.planner.generate_plan(test_question)
            health['planner_healthy'] = bool(plan and len(plan) >= 2)

            # Test executor
            if health['planner_healthy']:
                execution = self.executor.execute_step(
                    test_question,
                    "Calculate the total number of apples",
                    ""
                )
                health['executor_healthy'] = bool(execution and execution.get('raw_output'))

            # Test critic
            if health['executor_healthy']:
                evaluation = self.critic.evaluate_step(
                    test_question,
                    "Calculate the total number of apples",
                    "John has 5 + 3 = 8 apples total.",
                    ""
                )
                health['critic_healthy'] = 'is_correct' in evaluation

            health['system_healthy'] = all([
                health['planner_healthy'],
                health['executor_healthy'],
                health['critic_healthy']
            ])

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

        return health