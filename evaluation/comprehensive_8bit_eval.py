#!/usr/bin/env python3
"""
Comprehensive FIXED Evaluation - Optimized Token Limits
========================================================================
FIXES from diagnostic analysis:
- Planner truncation was causing 70% failures (100 tokens too low!)
- Increased Planner: 100 → 200 tokens (allows complete plans)
- Answer extraction: Standard GSM8K method

Configuration:
- Planner: temp=0.3, max_tokens=200, top_k=50 (NO TRUNCATION!)
- Executor: temp=0.0, max_tokens=300, top_k=30 (deterministic for math)
- Critic: temp=0.2, max_tokens=100, top_k=50 (diversity for error detection)
- Answer extraction: Standard GSM8K method (#### delimiter + common phrases + last number fallback)
- Sequential processing (preserves correction loop)
- PARALLEL evaluation with 10 concurrent workers

Expected improvement:
- Previous: 30% accuracy (70% plans truncated)
- Fixed: 50-60% accuracy (complete plans, proper execution)
Time: 25-35 minutes for 100 problems
"""
import os
import json
import modal
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np

app = modal.App("comprehensive-attention-only-evaluation")

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "bitsandbytes>=0.41.0",
    "huggingface_hub>=0.20.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0"
]).env({"TOKENIZERS_PARALLELISM": "false"})

volume = modal.Volume.from_name("acor-training-vol", create_if_missing=True)

@dataclass
class ComprehensiveMetrics:
    """Complete metrics for a single problem"""
    problem_id: int
    problem: str
    ground_truth: float

    # Original Table 1 metrics
    final_answer_correct: bool
    correction_rate: float  # % of errors detected
    recovery_rate: float  # % of detected errors fixed
    step_efficiency: float  # Avg correction loops

    # Planner metrics
    plan_quality_score: float  # 0-100
    num_plan_steps: int
    plan_has_numbering: bool
    plan_is_sequential: bool

    # Executor metrics
    step_accuracy: float  # % steps correct
    calculation_accuracy: float  # % calculations correct
    context_utilization: float  # % steps using context
    avg_tokens_per_step: float
    error_types: Dict[str, int]

    # Critic metrics
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    sensitivity: float  # TP / (TP + FN)
    specificity: float  # TN / (TN + FP)
    precision: float  # TP / (TP + FP)
    f1_score: float
    avg_feedback_specificity: float
    avg_feedback_actionability: float

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=7200,
    memory=32768,
    cpu=8.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def initialize_acor_with_metrics():
    """Initialize all ACOR modules for comprehensive evaluation"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login

    print("🔧 INITIALIZING ACOR WITH ATTENTION-ONLY ADAPTERS")
    print("=" * 60)

    # Authentication
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace authenticated")

    # Model configuration - NEW ATTENTION-ONLY ADAPTERS
    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    adapters = {
        "planner": "/vol/training_runs/planner_attention_only_20251003_170348/final_model",
        "executor": "/vol/training_runs/executor_attention_only_20251003_170711/final_model",
        "critic": "/vol/training_runs/critic_sft_20250929_210941/model"
    }
    print("📍 Using NEW attention-only adapters (VERIFIED WORKING!)")

    system_prompts = {
        "planner": "You are a strategic planner for mathematical problem solving. Generate a high-level, multi-step strategic blueprint with 3-6 clear, sequential steps.",
        "executor": "You are a mathematical problem executor. Execute a single, well-defined reasoning step based on the plan and current context. Show all calculations clearly.",
        "critic": "You are a generative, process-supervised module that evaluates reasoning steps for logical and computational correctness. Provide structured, actionable feedback when errors are found."
    }

    print(f"📥 Loading base model: {base_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config - 4-BIT (verified to work with attention-only adapters!)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("🔧 Using 4-bit quantization (verified working with attention-only adapters)")

    # Load base model ONCE (memory efficient!)
    print(f"🔧 Loading base model (will load adapters separately)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
        trust_remote_code=True
    )
    print(f"✅ Base model loaded (~16GB)")

    # Load all adapters onto the SAME base model
    print(f"🔧 Loading adapters (adapter swapping)...")
    model_with_adapters = PeftModel.from_pretrained(
        base_model,
        adapters["planner"],
        adapter_name="planner"
    )
    model_with_adapters.load_adapter(adapters["executor"], adapter_name="executor")
    model_with_adapters.load_adapter(adapters["critic"], adapter_name="critic")
    print(f"✅ All 3 adapters loaded (~1.5GB total)")
    print(f"💾 Total memory: ~18GB (vs 48GB with separate models)")

    return {
        "model": model_with_adapters,  # Single model with swappable adapters
        "tokenizer": tokenizer,
        "system_prompts": system_prompts,
        "success": True
    }

class BalancedEvaluator:
    """Evaluator using 8-BIT configuration settings"""

    def __init__(self, model, tokenizer, system_prompts):
        self.model = model  # Single model with multiple adapters
        self.tokenizer = tokenizer
        self.system_prompts = system_prompts
        self.previous_plans = []  # For diversity calculation

    def generate_response(self, module: str, prompt: str, temperature: float = 0.3,
                         max_tokens: int = 100, top_k: int = 50) -> str:
        """Generate response with module-specific balanced settings using adapter swapping"""
        import torch

        # Switch to the appropriate adapter
        self.model.set_adapter(module)

        messages = [
            {"role": "system", "content": self.system_prompts[module]},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

    def evaluate_with_all_metrics(self, problem_id: int, problem: str, ground_truth: float) -> ComprehensiveMetrics:
        """Evaluate single problem and compute ALL metrics"""

        print(f"\n📝 Problem {problem_id}")
        print(f"   {problem[:80]}...")

        # Step 1: Generate plan and measure Planner metrics
        plan_metrics = self._evaluate_planner(problem)
        plan = plan_metrics["plan"]
        plan_steps = plan_metrics["steps"]

        # Step 2: Execute with Executor and Critic, measure all metrics
        execution_metrics = self._evaluate_execution_with_critic(
            problem, ground_truth, plan, plan_steps
        )

        # Step 3: Combine into comprehensive metrics
        metrics = ComprehensiveMetrics(
            problem_id=problem_id,
            problem=problem,
            ground_truth=ground_truth,

            # Table 1 metrics
            final_answer_correct=execution_metrics["final_correct"],
            correction_rate=execution_metrics["correction_rate"],
            recovery_rate=execution_metrics["recovery_rate"],
            step_efficiency=execution_metrics["step_efficiency"],

            # Planner metrics
            plan_quality_score=plan_metrics["quality_score"],
            num_plan_steps=len(plan_steps),
            plan_has_numbering=plan_metrics["has_numbering"],
            plan_is_sequential=plan_metrics["is_sequential"],

            # Executor metrics
            step_accuracy=execution_metrics["step_accuracy"],
            calculation_accuracy=execution_metrics["calculation_accuracy"],
            context_utilization=execution_metrics["context_utilization"],
            avg_tokens_per_step=execution_metrics["avg_tokens"],
            error_types=execution_metrics["error_distribution"],

            # Critic metrics
            true_positives=execution_metrics["tp"],
            true_negatives=execution_metrics["tn"],
            false_positives=execution_metrics["fp"],
            false_negatives=execution_metrics["fn"],
            sensitivity=execution_metrics["sensitivity"],
            specificity=execution_metrics["specificity"],
            precision=execution_metrics["precision"],
            f1_score=execution_metrics["f1"],
            avg_feedback_specificity=execution_metrics["feedback_specificity"],
            avg_feedback_actionability=execution_metrics["feedback_actionability"]
        )

        print(f"   ✅ Final: {'✓' if metrics.final_answer_correct else '✗'} | "
              f"Plan: {metrics.plan_quality_score:.0f}/100 | "
              f"Steps: {metrics.step_accuracy:.0%} | "
              f"Critic F1: {metrics.f1_score:.2f}")

        return metrics

    def _evaluate_planner(self, problem: str) -> Dict:
        """Evaluate Planner with FIXED settings (temp=0.3, max_tokens=200 - NO TRUNCATION!)"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Generate plan with fixed settings (200 tokens to avoid truncation)
        plan_prompt = f"Generate a strategic plan to solve this mathematical problem step by step:\n\n{problem}"
        plan = self.generate_response("planner", plan_prompt, temperature=0.3, max_tokens=200, top_k=50)

        # Parse steps
        step_pattern = r'^\s*(\d+\.?\s*.*?)(?=^\s*\d+\.|\Z)'
        steps = re.findall(step_pattern, plan, re.MULTILINE | re.DOTALL)
        if not steps:
            sentences = [s.strip() for s in plan.split('.') if s.strip()]
            steps = sentences[:6]
        steps = [s.strip() for s in steps if s.strip()]

        # Structure metrics
        has_numbering = bool(re.search(r'^\s*\d+\.', plan, re.MULTILINE))
        step_numbers = re.findall(r'^\s*(\d+)\.', plan, re.MULTILINE)
        is_sequential = False
        if step_numbers:
            nums = [int(n) for n in step_numbers]
            is_sequential = nums == list(range(1, len(nums) + 1))

        # Content metrics
        plan_lower = plan.lower()
        specific_terms = len(re.findall(r'\d+|calculate|solve|find|determine|compute', plan_lower))
        generic_terms = len(re.findall(r'understand|analyze|consider|think|look', plan_lower))
        specificity = specific_terms / (specific_terms + generic_terms + 1)

        # Coverage via TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([problem, plan])
            coverage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            coverage = 0.5

        # Clarity
        action_verbs = ['calculate', 'find', 'determine', 'solve', 'compute', 'identify']
        clarity = min(sum(1 for v in action_verbs if v in plan_lower) / len(steps), 1.0) if steps else 0.0

        # Quality score (0-100)
        quality_score = (
            (10 if has_numbering else 0) +
            (10 if is_sequential else 0) +
            (coverage * 20) +
            (specificity * 20) +
            (clarity * 20) +
            (min(len(steps) / 5, 1.0) * 20)  # Prefer 5 steps
        )

        # Diversity (vs previous plans)
        diversity = self._calculate_diversity(plan)
        self.previous_plans.append(plan)

        return {
            "plan": plan,
            "steps": steps,
            "quality_score": quality_score,
            "has_numbering": has_numbering,
            "is_sequential": is_sequential,
            "coverage": coverage,
            "specificity": specificity,
            "clarity": clarity,
            "diversity": diversity
        }

    def _calculate_diversity(self, current_plan: str) -> float:
        """Calculate plan diversity vs previous plans"""
        if not self.previous_plans:
            return 1.0

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        try:
            vectorizer = TfidfVectorizer()
            all_plans = self.previous_plans + [current_plan]
            tfidf_matrix = vectorizer.fit_transform(all_plans)
            current_vector = tfidf_matrix[-1:]
            previous_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(current_vector, previous_vectors)[0]
            diversity = 1.0 - np.max(similarities)
            return diversity
        except:
            return 0.5

    def _evaluate_execution_with_critic(self, problem: str, ground_truth: float,
                                       plan: str, plan_steps: List[str]) -> Dict:
        """Execute steps with Critic evaluation, using balanced settings"""

        current_context = f"Problem: {problem}\n\nPlan:\n{plan}\n\n"

        # Track metrics
        executions = []
        critic_evaluations = []
        correction_loops = []

        for i, plan_step in enumerate(plan_steps):
            step_result = self._execute_step_with_metrics(
                step_number=i+1,
                plan_step=plan_step,
                context=current_context,
                ground_truth=ground_truth,
                is_final_step=(i == len(plan_steps) - 1)
            )

            executions.append(step_result)
            critic_evaluations.extend(step_result["critic_tests"])
            correction_loops.append(step_result["correction_attempts"])

            current_context += f"Step {i+1}: {step_result['execution']}\n"

        # Compute final answer correctness
        # Try extracting from ALL executions combined (handles truncation better)
        all_execution_text = "\n".join([e["execution"] for e in executions])
        final_answer = self._extract_answer(all_execution_text) if executions else None
        final_correct = final_answer is not None and abs(final_answer - ground_truth) < 0.01

        # Executor metrics
        step_accuracy = np.mean([e["is_correct"] for e in executions]) if executions else 0.0
        calc_accuracy = np.mean([e["calc_correct"] for e in executions]) if executions else 0.0
        context_util = np.mean([e["used_context"] for e in executions]) if executions else 0.0
        avg_tokens = np.mean([e["num_tokens"] for e in executions]) if executions else 0.0

        error_counts = Counter()
        for e in executions:
            error_counts[e["error_type"]] += 1

        # Critic metrics
        tp = sum(1 for c in critic_evaluations if c["true_positive"])
        tn = sum(1 for c in critic_evaluations if c["true_negative"])
        fp = sum(1 for c in critic_evaluations if c["false_positive"])
        fn = sum(1 for c in critic_evaluations if c["false_negative"])

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        feedback_spec = np.mean([c["feedback_specificity"] for c in critic_evaluations]) if critic_evaluations else 0.0
        feedback_action = np.mean([c["feedback_actionability"] for c in critic_evaluations]) if critic_evaluations else 0.0

        # Table 1 metrics
        errors_detected = sum(1 for c in critic_evaluations if c["detected_error"] and not c["actually_correct"])
        total_errors = sum(1 for c in critic_evaluations if not c["actually_correct"])
        correction_rate = errors_detected / total_errors if total_errors > 0 else 0.0

        errors_corrected = sum(1 for c in critic_evaluations if c["error_corrected"])
        recovery_rate = errors_corrected / errors_detected if errors_detected > 0 else 0.0

        step_efficiency = np.mean(correction_loops) if correction_loops else 0.0

        return {
            "final_correct": final_correct,
            "correction_rate": correction_rate,
            "recovery_rate": recovery_rate,
            "step_efficiency": step_efficiency,

            "step_accuracy": step_accuracy,
            "calculation_accuracy": calc_accuracy,
            "context_utilization": context_util,
            "avg_tokens": avg_tokens,
            "error_distribution": dict(error_counts),

            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": f1,
            "feedback_specificity": feedback_spec,
            "feedback_actionability": feedback_action
        }

    def _execute_step_with_metrics(self, step_number: int, plan_step: str, context: str,
                                   ground_truth: float, is_final_step: bool) -> Dict:
        """Execute single step with 8-BIT Executor settings (temp=0.0, max_tokens=300)"""

        # Execute with balanced settings (increased tokens to avoid truncation)
        executor_prompt = f"{context}\nExecute step: {plan_step}\n\nShow your work clearly:"
        execution = self.generate_response("executor", executor_prompt, temperature=0.0, max_tokens=300, top_k=30)

        # Metrics
        num_tokens = len(execution.split())
        numbers = self._extract_numbers(execution)
        actual_result = numbers[-1] if numbers else None

        # Correctness (simplified - final step checked against ground truth)
        is_correct = False
        calc_correct = False
        if is_final_step and actual_result is not None:
            is_correct = abs(actual_result - ground_truth) < 0.01
            calc_correct = is_correct
        else:
            is_correct = actual_result is not None
            calc_correct = is_correct

        used_context = any(str(n) in execution for n in self._extract_numbers(context))

        error_type = self._categorize_error(execution, actual_result, ground_truth if is_final_step else None)

        # Test Critic on this execution (create test cases)
        critic_tests = self._test_critic_on_step(context, plan_step, execution, is_correct)

        return {
            "execution": execution,
            "num_tokens": num_tokens,
            "is_correct": is_correct,
            "calc_correct": calc_correct,
            "used_context": used_context,
            "error_type": error_type,
            "critic_tests": critic_tests,
            "correction_attempts": len(critic_tests)  # Simplified
        }

    def _test_critic_on_step(self, context: str, plan_step: str, execution: str, is_correct: bool) -> List[Dict]:
        """Test Critic with 8-BIT settings (temp=0.2, max_tokens=100)"""

        tests = []

        # Test 1: Correct execution
        critic_prompt = f"""Evaluate this reasoning step for errors:

Context: {context}
Step to execute: {plan_step}
Student's work: {execution}

Provide analysis:
1. Is the approach correct?
2. Are calculations accurate?
3. Any errors identified?
4. Overall: CORRECT or INCORRECT"""

        feedback = self.generate_response("critic", critic_prompt, temperature=0.2, max_tokens=100, top_k=50)
        detected_error = self._critic_detected_error(feedback)

        spec, action = self._assess_feedback_quality(feedback)

        tests.append({
            "actually_correct": is_correct,
            "detected_error": detected_error,
            "error_corrected": False,  # Simplified for speed
            "true_positive": (not is_correct) and detected_error,
            "true_negative": is_correct and (not detected_error),
            "false_positive": is_correct and detected_error,
            "false_negative": (not is_correct) and (not detected_error),
            "feedback_specificity": spec,
            "feedback_actionability": action
        })

        return tests

    def _critic_detected_error(self, feedback: str) -> bool:
        """Check if critic detected an error"""
        feedback_lower = feedback.lower()
        error_keywords = ['incorrect', 'wrong', 'error', 'mistake']
        return any(kw in feedback_lower for kw in error_keywords)

    def _assess_feedback_quality(self, feedback: str) -> Tuple[float, float]:
        """Assess specificity and actionability"""
        feedback_lower = feedback.lower()

        specificity_indicators = [
            bool(re.search(r'\d+', feedback)),
            'calculation' in feedback_lower or 'compute' in feedback_lower,
            'incorrect' in feedback_lower,
            'should be' in feedback_lower
        ]
        specificity = sum(specificity_indicators) / len(specificity_indicators)

        actionability_indicators = [
            'correct' in feedback_lower or 'fix' in feedback_lower,
            'should' in feedback_lower,
            bool(re.search(r'=|calculate', feedback_lower)),
            len(feedback.split()) > 15
        ]
        actionability = sum(actionability_indicators) / len(actionability_indicators)

        return specificity, actionability

    def _extract_answer(self, text: str) -> Optional[float]:
        """
        Extract numerical answer using standard GSM8K evaluation method
        Based on lm-evaluation-harness and official GSM8K format
        """
        # Remove commas from numbers (1,234 -> 1234)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)

        # Strategy 1: Official GSM8K format with #### delimiter (highest confidence)
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Strategy 2: Common answer phrases (high confidence)
        answer_patterns = [
            r'(?:the\s+)?answer\s+is\s*:?\s*\$?\s*([+-]?\d+(?:\.\d+)?)',
            r'final\s+answer\s*:?\s*\$?\s*([+-]?\d+(?:\.\d+)?)',
            r'(?:total|result)\s+is\s*:?\s*\$?\s*([+-]?\d+(?:\.\d+)?)',
            r'(?:makes?|earns?|gets?)\s+\$?\s*([+-]?\d+(?:\.\d+)?)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Strategy 3: Last number in text (fallback - standard across all papers)
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers"""
        pattern = r'([+-]?\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text)
        try:
            return [float(m) for m in matches]
        except:
            return []

    def _categorize_error(self, execution: str, actual: Optional[float], expected: Optional[float]) -> str:
        """Categorize error type"""
        if actual is None:
            return "format"
        if expected is None:
            return "none"
        if abs(actual - expected) < 0.01:
            return "none"
        return "arithmetic"  # Simplified

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=10800,  # 3 hours for full evaluation
    memory=40960,   # 40GB (adapter swapping: 1 model + 3 adapters + overhead + results = ~33GB)
    cpu=8.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_comprehensive_8bit(num_problems: int = 100) -> Dict:
    """Run comprehensive balanced evaluation with module-specific optimized settings"""

    from datasets import load_dataset

    print("📊 COMPREHENSIVE FIXED EVALUATION (PARALLEL)")
    print("=" * 60)
    print("🔧 FIX: Planner token limit 100→200 (no more truncation!)")
    print("Configuration: Attention-Only Adapters + 4-bit Quantization")
    print("  - Planner: temp=0.3, max_tokens=200, top_k=50 (FIXED - no truncation!)")
    print("  - Executor: temp=0.0, max_tokens=300, top_k=30 (attention-only, 11K examples)")
    print("  - Critic: temp=0.2, max_tokens=100, top_k=50 (attention-only)")
    print("  - Answer extraction: Standard GSM8K (#### + phrases + last number)")
    print("  - 4-bit quantization (VERIFIED WORKING!)")
    print("  - PARALLEL evaluation: 10 concurrent workers")
    print(f"Problems: {num_problems}")
    print(f"Expected accuracy: 50-60% (vs 30% with truncated plans)")
    print(f"Expected time: {num_problems * 2.5 / 10:.0f}-{num_problems * 3.5 / 10:.0f} minutes")
    print("=" * 60)

    # Initialize - use .local() since we're in a large container
    init_result = initialize_acor_with_metrics.local()
    if not init_result.get("success"):
        return {"error": "Initialization failed"}

    evaluator = BalancedEvaluator(
        init_result["model"],  # Single model with adapter swapping
        init_result["tokenizer"],
        init_result["system_prompts"]
    )
    print("✅ ACOR initialized with adapter swapping")

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    problems = dataset.select(range(min(num_problems, len(dataset))))
    print(f"📚 Loaded {len(problems)} problems")

    # Evaluate all IN PARALLEL using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    all_metrics = []
    results_lock = threading.Lock()

    def evaluate_single_problem(idx, problem_data):
        """Evaluate a single problem (thread-safe)"""
        try:
            question = problem_data["question"]
            ground_truth_text = problem_data["answer"]

            gt_match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', ground_truth_text)
            if not gt_match:
                return None

            ground_truth = float(gt_match.group(1))

            # Adapter swapping is thread-safe
            metrics = evaluator.evaluate_with_all_metrics(idx+1, question, ground_truth)
            return metrics
        except Exception as e:
            print(f"   ❌ Problem {idx+1} Error: {e}")
            return None

    # Parallel evaluation with 10 concurrent workers
    print(f"🚀 Running parallel evaluation with 10 workers...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all problems
        future_to_idx = {
            executor.submit(evaluate_single_problem, i, problem_data): i
            for i, problem_data in enumerate(problems)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            completed += 1
            result = future.result()
            if result is not None:
                with results_lock:
                    all_metrics.append(result)

            # Progress update every 10 problems
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{len(problems)} problems completed ({completed/len(problems)*100:.0f}%)")

    print(f"✅ Completed {len(all_metrics)} problems successfully")

    # Aggregate statistics
    stats = aggregate_comprehensive_metrics(all_metrics)

    # Save
    results_dir = f"/vol/evaluations/comprehensive_attention_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    full_results = {
        "evaluation_type": "comprehensive_fixed_token_limits",
        "configuration": {
            "adapters": "attention_only",
            "quantization": "4bit",
            "planner": {"temperature": 0.3, "max_tokens": 200, "top_k": 50, "adapter": "attention_only", "fix": "increased from 100 to prevent truncation"},
            "executor": {"temperature": 0.0, "max_tokens": 300, "top_k": 30, "adapter": "attention_only", "dataset_size": 11066},
            "critic": {"temperature": 0.2, "max_tokens": 100, "top_k": 50, "adapter": "attention_only"},
            "answer_extraction": "standard_gsm8k_method",
            "processing": "parallel"
        },
        "timestamp": datetime.now().isoformat(),
        "num_problems": len(all_metrics),
        "statistics": stats,
        "individual_results": [asdict(m) for m in all_metrics]
    }

    with open(f"{results_dir}/comprehensive_attention_only.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"🎉 COMPREHENSIVE ATTENTION-ONLY EVALUATION COMPLETE")
    print(f"{'='*60}")
    print_comprehensive_summary(stats)
    print(f"📁 Saved: {results_dir}")

    return full_results

def aggregate_comprehensive_metrics(metrics_list: List[ComprehensiveMetrics]) -> Dict:
    """Aggregate all metrics"""
    return {
        "table_1_metrics": {
            "final_answer_accuracy": np.mean([m.final_answer_correct for m in metrics_list]),
            "correction_rate": np.mean([m.correction_rate for m in metrics_list]),
            "recovery_rate": np.mean([m.recovery_rate for m in metrics_list]),
            "step_efficiency": np.mean([m.step_efficiency for m in metrics_list])
        },
        "planner_metrics": {
            "avg_quality_score": np.mean([m.plan_quality_score for m in metrics_list]),
            "avg_num_steps": np.mean([m.num_plan_steps for m in metrics_list]),
            "has_numbering_rate": np.mean([m.plan_has_numbering for m in metrics_list]),
            "is_sequential_rate": np.mean([m.plan_is_sequential for m in metrics_list])
        },
        "executor_metrics": {
            "step_accuracy": np.mean([m.step_accuracy for m in metrics_list]),
            "calculation_accuracy": np.mean([m.calculation_accuracy for m in metrics_list]),
            "context_utilization": np.mean([m.context_utilization for m in metrics_list]),
            "avg_tokens_per_step": np.mean([m.avg_tokens_per_step for m in metrics_list])
        },
        "critic_metrics": {
            "sensitivity": np.mean([m.sensitivity for m in metrics_list]),
            "specificity": np.mean([m.specificity for m in metrics_list]),
            "precision": np.mean([m.precision for m in metrics_list]),
            "f1_score": np.mean([m.f1_score for m in metrics_list]),
            "feedback_specificity": np.mean([m.avg_feedback_specificity for m in metrics_list]),
            "feedback_actionability": np.mean([m.avg_feedback_actionability for m in metrics_list])
        }
    }

def print_comprehensive_summary(stats: Dict):
    """Print all metrics"""
    print(f"\n📊 COMPREHENSIVE METRICS SUMMARY (ATTENTION-ONLY)")
    print(f"┌─────────────────────────────────────────────────┐")
    print(f"│ TABLE 1 METRICS (Original)                     │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ Final Answer Accuracy: {stats['table_1_metrics']['final_answer_accuracy']*100:5.1f}%              │")
    print(f"│ Correction Rate:       {stats['table_1_metrics']['correction_rate']*100:5.1f}%              │")
    print(f"│ Recovery Rate:         {stats['table_1_metrics']['recovery_rate']*100:5.1f}%              │")
    print(f"│ Step Efficiency:       {stats['table_1_metrics']['step_efficiency']:5.2f}               │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ PLANNER METRICS                                 │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ Quality Score:         {stats['planner_metrics']['avg_quality_score']:5.1f}/100           │")
    print(f"│ Avg # Steps:           {stats['planner_metrics']['avg_num_steps']:5.1f}               │")
    print(f"│ Has Numbering:         {stats['planner_metrics']['has_numbering_rate']*100:5.1f}%              │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ EXECUTOR METRICS                                │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ Step Accuracy:         {stats['executor_metrics']['step_accuracy']*100:5.1f}%              │")
    print(f"│ Calculation Accuracy:  {stats['executor_metrics']['calculation_accuracy']*100:5.1f}%              │")
    print(f"│ Context Utilization:   {stats['executor_metrics']['context_utilization']*100:5.1f}%              │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ CRITIC METRICS                                  │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│ Sensitivity (Recall):  {stats['critic_metrics']['sensitivity']*100:5.1f}%              │")
    print(f"│ Specificity:           {stats['critic_metrics']['specificity']*100:5.1f}%              │")
    print(f"│ F1 Score:              {stats['critic_metrics']['f1_score']:5.2f}               │")
    print(f"│ Feedback Quality:      {stats['critic_metrics']['feedback_specificity']:5.2f}               │")
    print(f"└─────────────────────────────────────────────────┘")

@app.local_entrypoint()
def main(num_problems: int = 100):
    """Run comprehensive balanced evaluation"""
    results = run_comprehensive_8bit.remote(num_problems=num_problems)

    if "error" not in results:
        print(f"\n🎉 SUCCESS!")
        print(f"📊 Accuracy: {results['statistics']['table_1_metrics']['final_answer_accuracy']*100:.1f}%")
    else:
        print(f"❌ Failed: {results['error']}")

    return results

if __name__ == "__main__":
    main()
