#!/usr/bin/env python3
"""
Test Attention-Only Adapters with 4-bit Quantization
=====================================================
Verify that attention-only adapters (q_proj, k_proj, v_proj, o_proj)
work correctly with 4-bit quantization at inference time.

This should work (unlike MLP-inclusive adapters which get bypassed).
"""
import os
import json
import modal
from typing import Dict

app = modal.App("test-attention-only-adapters")

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
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
def test_attention_only_adapters() -> Dict:
    """Test both Planner and Executor attention-only adapters"""

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login

    print("="*60)
    print("🔬 TESTING ATTENTION-ONLY ADAPTERS")
    print("="*60)

    # Auth
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace authenticated")

    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # New attention-only adapters
    adapters = {
        "planner": "/vol/training_runs/planner_attention_only_20251003_170348/final_model",
        "executor": "/vol/training_runs/executor_attention_only_20251003_170711/final_model"
    }

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config (same as inference)
    print("\n🔧 Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    results = {}

    # Test each adapter
    for adapter_name, adapter_path in adapters.items():
        print(f"\n{'='*60}")
        print(f"Testing {adapter_name.upper()} (Attention-Only)")
        print(f"{'='*60}")

        # Load base model with 4-bit quantization
        print(f"📥 Loading base model with 4-bit quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True
        )

        # Test 1: Base model output (no adapter)
        print(f"\n🧪 Test 1: Base model (no adapter)")
        test_prompt = "Calculate step by step: What is 2 + 2?"

        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": test_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        base_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"📄 Base output ({len(base_output)} chars):")
        print(f"   {base_output[:200]}...")

        # Test 2: With attention-only adapter
        print(f"\n🧪 Test 2: With attention-only adapter")
        print(f"📂 Loading adapter from {adapter_path}...")

        model_with_adapter = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=adapter_name
        )
        model_with_adapter.set_adapter(adapter_name)
        print(f"✅ Adapter loaded and activated")

        # Generate with adapter
        with torch.no_grad():
            outputs = model_with_adapter.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        adapter_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"📄 Adapter output ({len(adapter_output)} chars):")
        print(f"   {adapter_output[:200]}...")

        # Compare
        outputs_identical = (base_output == adapter_output)
        print(f"\n{'='*60}")
        if outputs_identical:
            print("❌ FAIL: Outputs are IDENTICAL (adapter not applied!)")
            verdict = "FAILED"
        else:
            print("✅ PASS: Outputs are DIFFERENT (adapter working!)")
            verdict = "PASSED"
        print(f"{'='*60}")

        results[adapter_name] = {
            "verdict": verdict,
            "base_output": base_output,
            "adapter_output": adapter_output,
            "outputs_identical": outputs_identical,
            "base_length": len(base_output),
            "adapter_length": len(adapter_output)
        }

        # Clean up
        del base_model
        del model_with_adapter
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")

    all_passed = all(r["verdict"] == "PASSED" for r in results.values())

    for adapter_name, result in results.items():
        status = "✅" if result["verdict"] == "PASSED" else "❌"
        print(f"{status} {adapter_name.upper()}: {result['verdict']}")
        print(f"   Base: {result['base_length']} chars")
        print(f"   Adapter: {result['adapter_length']} chars")
        print(f"   Identical: {result['outputs_identical']}")

    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - Attention-only adapters work!")
    else:
        print("❌ SOME TESTS FAILED - Investigation needed")
    print(f"{'='*60}")

    return {
        "all_passed": all_passed,
        "results": results
    }

@app.local_entrypoint()
def main():
    """Run attention-only adapter tests"""
    print("🚀 Testing attention-only adapters with 4-bit quantization")
    print("   Expected: Adapters should work (outputs should differ from base)")
    print()

    result = test_attention_only_adapters.remote()

    print("\n📋 TEST RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)

    if result["all_passed"]:
        print("\n✅ SUCCESS! Attention-only adapters work with 4-bit quantization")
        print("🎯 Next step: Run full evaluation with these adapters")
    else:
        print("\n❌ FAILED: Attention-only adapters still have issues")
        print("🔍 Need to investigate further")

if __name__ == "__main__":
    main()
