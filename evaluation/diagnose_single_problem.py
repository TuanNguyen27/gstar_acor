#!/usr/bin/env python3
"""
Diagnose Single Problem - See Actual Model Outputs
===================================================
Run a single GSM8K problem through ACOR and show detailed outputs
to understand where failures occur.
"""
import os
import json
import modal

app = modal.App("diagnose-single-problem")

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "bitsandbytes>=0.41.0",
    "huggingface_hub>=0.20.0",
]).env({"TOKENIZERS_PARALLELISM": "false"})

volume = modal.Volume.from_name("acor-training-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=600,
    memory=32768,
    cpu=8.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def diagnose_problem(problem_text: str, ground_truth: float):
    """Run single problem and show all intermediate outputs"""

    import torch
    import re
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login

    print("="*80)
    print("🔍 DIAGNOSING SINGLE PROBLEM")
    print("="*80)

    # Auth
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)

    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    adapters = {
        "planner": "/vol/training_runs/planner_attention_only_20251003_170348/final_model",
        "executor": "/vol/training_runs/executor_attention_only_20251003_170711/final_model",
        "critic": "/vol/training_runs/critic_sft_20250929_210941/model"
    }

    system_prompts = {
        "planner": "You are a strategic planner for mathematical problem solving. Generate a high-level, multi-step strategic blueprint with 3-6 clear, sequential steps.",
        "executor": "You are a mathematical problem executor. Execute a single, well-defined reasoning step based on the plan and current context. Show all calculations clearly.",
        "critic": "You are a generative, process-supervised module that evaluates reasoning steps for logical and computational correctness. Provide structured, actionable feedback when errors are found."
    }

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("📥 Loading base model with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )

    print("🔧 Loading adapters...")
    model = PeftModel.from_pretrained(base_model, adapters["planner"], adapter_name="planner")
    model.load_adapter(adapters["executor"], adapter_name="executor")
    model.load_adapter(adapters["critic"], adapter_name="critic")
    print("✅ All adapters loaded")

    print("\n" + "="*80)
    print(f"📝 PROBLEM: {problem_text}")
    print(f"🎯 GROUND TRUTH: {ground_truth}")
    print("="*80)

    # STEP 1: PLANNER
    print("\n" + "="*80)
    print("🗺️  STEP 1: PLANNER (FIXED: 200 tokens, was 100)")
    print("="*80)

    model.set_adapter("planner")
    planner_prompt = f"{system_prompts['planner']}\n\nProblem: {problem_text}\n\nGenerate a strategic plan:"
    messages = [{"role": "user", "content": planner_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # FIXED: was 100, caused truncation!
            temperature=0.3,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    plan = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n📋 PLAN GENERATED:\n{plan}\n")

    # Parse plan steps
    step_pattern = r'^\\s*(\\d+\\.?\\s*.*?)(?=^\\s*\\d+\\.|\\Z)'
    steps = re.findall(step_pattern, plan, re.MULTILINE | re.DOTALL)
    if not steps:
        sentences = [s.strip() for s in plan.split('.') if s.strip()]
        steps = sentences[:6]
    steps = [s.strip() for s in steps if s.strip()]

    print(f"📊 PARSED STEPS ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step[:80]}...")

    # STEP 2: EXECUTOR (on each step)
    print("\n" + "="*80)
    print("⚙️  STEP 2: EXECUTOR")
    print("="*80)

    model.set_adapter("executor")
    context = f"Problem: {problem_text}\n\nPlan:\n{plan}\n\n"
    all_executions = []

    for i, plan_step in enumerate(steps[:3], 1):  # Only first 3 steps for speed
        print(f"\n--- Executing Step {i}/{len(steps)} ---")
        print(f"Plan step: {plan_step[:80]}...")

        executor_prompt = f"{system_prompts['executor']}\n\n{context}\nExecute step: {plan_step}\n\nShow your work clearly:"
        messages = [{"role": "user", "content": executor_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        execution = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        all_executions.append(execution)

        print(f"🔧 EXECUTION:\n{execution}\n")

        context += f"Step {i}: {execution}\n"

    # STEP 3: EXTRACT FINAL ANSWER
    print("\n" + "="*80)
    print("🎯 STEP 3: EXTRACT FINAL ANSWER (Standard GSM8K Method)")
    print("="*80)

    # Combine all executions for extraction (handles truncation)
    all_execution_text = "\n".join(all_executions) if all_executions else ""

    # Standard GSM8K answer extraction (from lm-evaluation-harness)
    def extract_answer_standard_gsm8k(text: str) -> float:
        """Extract answer using standard GSM8K evaluation method"""
        # Remove commas from numbers
        text = re.sub(r"(\d),(\d)", r"\1\2", text)

        # Strategy 1: #### delimiter (official GSM8K format)
        match = re.search(r'####\s*([+-]?\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Strategy 2: Common answer phrases
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

        # Strategy 3: Last number (fallback)
        numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass

        return None

    predicted = extract_answer_standard_gsm8k(all_execution_text)

    # Show extraction details
    all_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', all_execution_text)
    print(f"\n📊 All numbers found: {all_numbers}")
    print(f"🔍 Extraction strategies tested:")
    print(f"   1. #### delimiter: {'✓' if '####' in all_execution_text else '✗'}")
    print(f"   2. Answer phrases: {'✓' if any(p in all_execution_text.lower() for p in ['answer is', 'makes', 'earns']) else '✗'}")
    print(f"   3. Last number fallback: {all_numbers[-1] if all_numbers else 'None'}")

    print(f"\n🔢 PREDICTED: {predicted}")
    print(f"🎯 GROUND TRUTH: {ground_truth}")

    correct = predicted is not None and abs(predicted - ground_truth) < 0.01
    print(f"\n{'✅ CORRECT!' if correct else '❌ WRONG!'}")

    # STEP 4: CRITIC EVALUATION
    print("\n" + "="*80)
    print("🔍 STEP 4: CRITIC")
    print("="*80)

    model.set_adapter("critic")
    critic_prompt = f"""{system_prompts['critic']}

Problem: {problem_text}

Solution: {all_executions[-1] if all_executions else 'No execution'}

Evaluate for errors:"""

    messages = [{"role": "user", "content": critic_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    critique = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n📝 CRITIQUE:\n{critique}\n")

    # SUMMARY
    print("\n" + "="*80)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"✅ Plan generated: {len(plan)} chars, {len(steps)} steps")
    print(f"✅ Executions: {len(all_executions)} steps executed")
    print(f"{'✅' if predicted is not None else '❌'} Answer extracted: {predicted}")
    print(f"{'✅' if correct else '❌'} Final result: {'CORRECT' if correct else 'WRONG'}")
    print(f"📉 Error: {abs(predicted - ground_truth) if predicted else 'N/A'}")
    print("="*80)

    return {
        "problem": problem_text,
        "ground_truth": ground_truth,
        "plan": plan,
        "steps": steps,
        "executions": all_executions,
        "predicted": predicted,
        "correct": correct,
        "critique": critique
    }

@app.local_entrypoint()
def main(problem_id: int = 0):
    """Diagnose a single problem"""

    # Problem selection
    problems = {
        0: {  # Janet's ducks (baseline test)
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "ground_truth": 18.0
        },
        25: {  # Kyle's book (100% steps, wrong answer)
            "question": "Kyle bought last year's best-selling book for $19.50. This is with a 25% discount from the original price. What was the original price of the book?",
            "ground_truth": 26.0
        },
        52: {  # Tom's ship (100% steps, wrong answer)
            "question": "Tom's ship can travel at 10 miles per hour. He is sailing from 1 to 4 PM. He then travels back at a rate of 6 mph. How long does it take him to get back?",
            "ground_truth": 5.0
        }
    }

    if problem_id not in problems:
        print(f"❌ Invalid problem_id: {problem_id}. Choose from: {list(problems.keys())}")
        return

    problem_data = problems[problem_id]
    problem = problem_data["question"]
    ground_truth = problem_data["ground_truth"]

    print(f"🚀 Running diagnostic on Problem {problem_id}...")
    print(f"Expected answer: {ground_truth}")
    print()

    result = diagnose_problem.remote(problem, ground_truth)

    print("\n" + "="*80)
    print("📋 DIAGNOSTIC COMPLETE")
    print("="*80)
    print(f"Result: {'✅ PASS' if result['correct'] else '❌ FAIL'}")

    if not result['correct']:
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"   - Plan quality: {len(result['plan'])} chars")
        print(f"   - Steps generated: {len(result['steps'])}")
        print(f"   - Executions completed: {len(result['executions'])}")
        print(f"   - Answer extracted: {result['predicted']}")
        print(f"   - Expected: {ground_truth}")

        if result['predicted'] is None:
            print("\n❌ PRIMARY ISSUE: Failed to extract numerical answer")
        elif len(result['executions']) == 0:
            print("\n❌ PRIMARY ISSUE: Executor produced no steps")
        elif len(result['executions']) < len(result['steps']):
            print("\n❌ PRIMARY ISSUE: Incomplete execution (some steps missing)")
        else:
            print("\n❌ PRIMARY ISSUE: Calculation error in execution")

if __name__ == "__main__":
    main()
