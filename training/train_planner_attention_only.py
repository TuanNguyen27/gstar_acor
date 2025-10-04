#!/usr/bin/env python3
"""
Planner Attention-Only LoRA Training
====================================
FIX: Remove MLP layers to enable 4-bit quantization compatibility
- Target modules: ATTENTION-ONLY (like Critic which works!)
- Same dataset (~9.7K examples)
- 2 epochs
- Should work with 4-bit quantization at inference
"""
import os
import json
import modal
from datetime import datetime
from typing import Dict

app = modal.App("planner-attention-only-training")

image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "bitsandbytes>=0.41.0",
    "huggingface_hub>=0.20.0",
    "trl>=0.8.0",
]).env({"TOKENIZERS_PARALLELISM": "false"})

volume = modal.Volume.from_name("acor-training-vol", create_if_missing=True)

# Planner Training Configuration
PLANNER_CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "dataset_path": "/vol/datasets/planner_full_sft_dataset.jsonl",
    "expected_dataset_size": 9691,

    # System prompt
    "system_prompt": "You are a strategic planner for mathematical problem solving. Generate a high-level, multi-step strategic blueprint with 3-6 clear, sequential steps.",

    # ATTENTION-ONLY LoRA Configuration (NO MLP layers!)
    "lora_config": {
        "target_modules": [
            # Attention layers ONLY (like Critic - proven to work!)
            "q_proj", "k_proj", "v_proj", "o_proj"
            # MLP layers REMOVED: gate_proj, up_proj, down_proj
        ],
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "use_dora": False
    },

    # Training Arguments
    "training_args": {
        "output_dir": "/vol/training_runs/planner_attention_only_{timestamp}",

        # 2 epochs
        "num_train_epochs": 2,

        # Batch configuration
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,  # Effective batch = 8

        # Learning rate
        "learning_rate": 3e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,

        # Evaluation and saving
        "eval_strategy": "steps",
        "eval_steps": 300,
        "save_strategy": "steps",
        "save_steps": 300,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 10,

        # Logging
        "logging_steps": 50,
        "report_to": "none",

        # Performance
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "bf16": True,
        "gradient_checkpointing": True,
    }
}

@app.function(
    image=image,
    gpu="L40S",
    volumes={"/vol": volume},
    timeout=30000,  # 8.3 hours
    memory=32768,
    cpu=8.0,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def train_planner() -> Dict:
    """Train Planner with regular LoRA"""

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import Dataset
    from huggingface_hub import login

    print("⚡ PLANNER LORA TRAINING")
    print("="*60)
    print("🎯 Training Configuration:")
    print("   📊 Dataset: 9,691 examples")
    print("   🔄 Epochs: 2")
    print("   ⏱️  Expected: 5-6 hours")
    print("   💾 GPU: L40S (48GB)")
    print("   🔧 LoRA: Regular (DoRA disabled)")
    print("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = datetime.now()

    # Authentication
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace authenticated")

    # Load and validate dataset
    print("📚 Loading Planner dataset...")
    try:
        with open(PLANNER_CONFIG["dataset_path"], "r") as f:
            data = [json.loads(line) for line in f]

        print(f"✅ Dataset loaded: {len(data)} samples")

        if len(data) != PLANNER_CONFIG["expected_dataset_size"]:
            print(f"⚠️  Warning: Expected {PLANNER_CONFIG['expected_dataset_size']} samples, got {len(data)}")

        # Validate data format
        sample = data[0]
        required_keys = ["instruction", "input", "output"]
        if not all(k in sample for k in required_keys):
            raise ValueError(f"Dataset missing required keys. Found: {sample.keys()}")

        print(f"📋 Sample instruction: {sample['instruction'][:100]}...")

    except Exception as e:
        return {
            "status": "failed",
            "error": f"Dataset loading failed: {str(e)}",
            "timestamp": timestamp
        }

    # Load model with quantization
    print("🤖 Loading Llama-3.1-8B-Instruct...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        PLANNER_CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        PLANNER_CONFIG["model_name"],
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply LoRA
    print("🔧 Applying LoRA configuration (DoRA disabled)...")
    lora_config = LoraConfig(**PLANNER_CONFIG["lora_config"])
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model configured: {trainable_params:,} / {total_params:,} params trainable ({100*trainable_params/total_params:.2f}%)")
    print(f"🎯 Target modules: {PLANNER_CONFIG['lora_config']['target_modules']}")

    # Prepare dataset
    def format_planner_example(example):
        """Format planner training example"""
        messages = [
            {"role": "system", "content": PLANNER_CONFIG["system_prompt"]},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    print("📝 Formatting dataset...")
    formatted_texts = [format_planner_example(ex) for ex in data]
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Split train/eval
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    print(f"📊 Split: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Calculate training stats
    steps_per_epoch = len(train_dataset) // (PLANNER_CONFIG["training_args"]["per_device_train_batch_size"] * PLANNER_CONFIG["training_args"]["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * PLANNER_CONFIG["training_args"]["num_train_epochs"]
    print(f"📈 Training: {steps_per_epoch} steps/epoch × {PLANNER_CONFIG['training_args']['num_train_epochs']} epochs = {total_steps} total steps")
    print(f"⏱️  Estimated time: {total_steps * 3.5 / 3600:.1f} hours (@ ~3.5s/step)")

    # Training arguments
    output_dir = PLANNER_CONFIG["training_args"]["output_dir"].format(timestamp=timestamp)
    training_args = TrainingArguments(
        output_dir=output_dir,
        **{k: v for k, v in PLANNER_CONFIG["training_args"].items() if k != "output_dir"}
    )

    # Trainer
    print("🚀 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    print("🏋️  Starting training...")
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    try:
        train_result = trainer.train()

        # Save final model EXPLICITLY
        print("💾 Saving final model...")
        final_model_path = f"{output_dir}/final_model"
        os.makedirs(final_model_path, exist_ok=True)

        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        volume.commit()
        print(f"✅ Final model saved to: {final_model_path}")

        # Verify
        if os.path.exists(f"{final_model_path}/adapter_config.json"):
            print("✅ Verified: adapter_config.json exists")
        else:
            print("⚠️  Warning: adapter_config.json not found!")

        volume.commit()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600

        # Training summary
        summary = {
            "status": "success",
            "timestamp": timestamp,
            "model_path": final_model_path,
            "dataset_size": len(data),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "total_steps": total_steps,
            "final_train_loss": train_result.training_loss,
            "duration_hours": round(duration, 2),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "gpu": "L40S",
            "config": PLANNER_CONFIG
        }

        # Save summary
        summary_path = f"{output_dir}/training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        volume.commit()

        print("="*60)
        print("✅ TRAINING COMPLETE!")
        print(f"⏱️  Duration: {duration:.2f} hours")
        print(f"📉 Final loss: {train_result.training_loss:.4f}")
        print(f"💾 Model saved: {final_model_path}")
        print("="*60)

        return summary

    except Exception as e:
        error_summary = {
            "status": "failed",
            "error": str(e),
            "timestamp": timestamp,
            "duration_hours": (datetime.now() - start_time).total_seconds() / 3600
        }
        print(f"❌ Training failed: {e}")
        return error_summary


@app.local_entrypoint()
def main():
    """Launch planner training

    IMPORTANT: Run with --detach flag:
    modal run --detach train_planner_lora.py
    """
    print("🚀 Launching Planner LoRA training (9.7K examples, 2 epochs)")
    print("⏱️  Expected duration: 5-6 hours")
    print("💰 Cost: ~$10 (L40S @ $1.50/hr)")
    print("⚠️  CRITICAL: Use 'modal run --detach' to prevent disconnection!")
    print()

    result = train_planner.remote()

    print("\n📋 TRAINING RESULT:")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)

    if result.get("status") == "success":
        print(f"✅ Training completed successfully in {result['duration_hours']:.2f} hours")
        print(f"💾 Model path: {result['model_path']}")
        print(f"📉 Final loss: {result['final_train_loss']:.4f}")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
