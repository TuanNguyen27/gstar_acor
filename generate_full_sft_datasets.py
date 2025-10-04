#!/usr/bin/env python3
"""
Full-scale ACOR SFT Data Generation Pipeline
Generates complete datasets with robust checkpointing
"""
import os
import sys

# Load API key from environment variable (set locally, not in code!)
# Usage: export LLM_GW_EXPRESS_KEY="your_api_key_here"
if "LLM_GW_EXPRESS_KEY" not in os.environ:
    raise ValueError("LLM_GW_EXPRESS_KEY environment variable not set. Please set it before running.")

import json
import time
from datetime import datetime
from pathlib import Path
import traceback
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_generation.gsm8k_parser.gsm8k_loader import GSM8KLoader
from data_generation.gsm8k_parser.planner_data_generator import PlannerDataGenerator
from data_generation.gsm8k_parser.executor_data_generator import ExecutorDataGenerator
from data_generation.synthetic_generator.critic_data_generator import CriticDataGenerator

class FullScaleACORPipeline:
    """Full-scale pipeline with robust checkpointing"""

    def __init__(self):
        self.output_dir = Path("full_sft_datasets")
        self.checkpoint_dir = Path("full_checkpoints")

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Full-scale configuration based on ACOR requirements
        self.config = {
            "gsm8k_size": 7500,  # Full GSM8K training set
            "planner_target": 7500,   # Base target from config
            "executor_target": 30000,  # 4x augmentation
            "critic_target": 20000,   # Diverse error patterns
            "batch_size": 50,         # Process in batches for checkpointing
            "checkpoint_interval": 100 # Save every 100 examples
        }

        self.progress = {
            "planner": {"generated": 0, "target": self.config["planner_target"]},
            "executor": {"generated": 0, "target": self.config["executor_target"]},
            "critic": {"generated": 0, "target": self.config["critic_target"]}
        }

    def load_checkpoint(self, module_name):
        """Load existing checkpoint"""
        checkpoint_file = self.checkpoint_dir / f"{module_name}_progress.json"

        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                self.progress[module_name]["generated"] = data.get("examples_generated", 0)
                print(f"📂 Loaded {module_name} checkpoint: {self.progress[module_name]['generated']} examples")
                return True
        return False

    def save_checkpoint(self, module_name, batch_data, append=True):
        """Save checkpoint and append to dataset file"""
        # Save progress
        checkpoint_data = {
            "module": module_name,
            "examples_generated": self.progress[module_name]["generated"],
            "target": self.progress[module_name]["target"],
            "last_updated": datetime.now().isoformat(),
            "config": self.config
        }

        checkpoint_file = self.checkpoint_dir / f"{module_name}_progress.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Append to dataset file
        dataset_file = self.output_dir / f"{module_name}_full_sft_dataset.jsonl"
        mode = 'a' if append else 'w'

        with open(dataset_file, mode) as f:
            for example in batch_data:
                f.write(json.dumps(example) + '\n')

        print(f"💾 Saved {len(batch_data)} examples to {dataset_file}")

    def generate_planner_full_dataset(self, gsm8k_examples):
        """Generate complete Planner dataset"""
        print(f"\n🎯 Generating Planner Dataset (Target: {self.config['planner_target']} examples)")

        # Load checkpoint
        self.load_checkpoint("planner")

        if self.progress["planner"]["generated"] >= self.config["planner_target"]:
            print(f"✅ Planner dataset already complete: {self.progress['planner']['generated']} examples")
            return

        try:
            generator = PlannerDataGenerator()
            remaining = self.config["planner_target"] - self.progress["planner"]["generated"]

            print(f"📝 Generating {remaining} remaining planner examples...")

            # Process in batches
            batch_size = self.config["batch_size"]
            num_batches = math.ceil(remaining / batch_size)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gsm8k_examples))
                batch_gsm8k = gsm8k_examples[start_idx:end_idx]

                print(f"  Batch {batch_idx + 1}/{num_batches}: Processing {len(batch_gsm8k)} GSM8K problems...")

                # Generate base examples
                base_examples = generator.generate_examples(batch_gsm8k, len(batch_gsm8k))

                if base_examples:
                    # Generate augmentations
                    augmented = generator.augment_with_variations(base_examples, variation_factor=1)
                    training_data = generator.create_training_format(augmented)

                    # Update progress
                    self.progress["planner"]["generated"] += len(training_data)

                    # Save checkpoint
                    append = batch_idx > 0 or self.progress["planner"]["generated"] > len(training_data)
                    self.save_checkpoint("planner", training_data, append=append)

                    print(f"    ✅ Generated {len(training_data)} examples (Total: {self.progress['planner']['generated']}/{self.config['planner_target']})")

                # Rate limiting
                time.sleep(3.0)

                # Check if we've reached target
                if self.progress["planner"]["generated"] >= self.config["planner_target"]:
                    break

            print(f"✅ Planner generation complete: {self.progress['planner']['generated']} examples")

        except Exception as e:
            print(f"❌ Planner generation failed: {e}")
            traceback.print_exc()

    def generate_executor_full_dataset(self, gsm8k_examples):
        """Generate complete Executor dataset"""
        print(f"\n🎯 Generating Executor Dataset (Target: {self.config['executor_target']} examples)")

        # Load checkpoint
        self.load_checkpoint("executor")

        if self.progress["executor"]["generated"] >= self.config["executor_target"]:
            print(f"✅ Executor dataset already complete: {self.progress['executor']['generated']} examples")
            return

        try:
            generator = ExecutorDataGenerator()
            remaining = self.config["executor_target"] - self.progress["executor"]["generated"]

            print(f"📝 Generating {remaining} remaining executor examples...")

            # Process in batches
            batch_size = self.config["batch_size"]
            examples_per_problem = 4  # Multiple steps per problem
            problems_per_batch = batch_size // examples_per_problem

            num_batches = math.ceil(len(gsm8k_examples) / problems_per_batch)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * problems_per_batch
                end_idx = min(start_idx + problems_per_batch, len(gsm8k_examples))
                batch_gsm8k = gsm8k_examples[start_idx:end_idx]

                print(f"  Batch {batch_idx + 1}/{num_batches}: Processing {len(batch_gsm8k)} GSM8K problems...")

                # Generate base examples (multiple per problem)
                target_examples = len(batch_gsm8k) * examples_per_problem
                base_examples = generator.generate_examples(batch_gsm8k, target_examples)

                if base_examples:
                    # Generate context variations
                    context_variations = generator.generate_context_variations(base_examples)

                    # Generate error correction examples
                    error_examples = generator.create_error_correction_examples(
                        base_examples[:len(base_examples)//4], error_rate=0.3
                    )

                    # Combine all examples
                    all_examples = context_variations + error_examples
                    training_data = generator.create_training_format(all_examples)

                    # Update progress
                    self.progress["executor"]["generated"] += len(training_data)

                    # Save checkpoint
                    append = batch_idx > 0 or self.progress["executor"]["generated"] > len(training_data)
                    self.save_checkpoint("executor", training_data, append=append)

                    print(f"    ✅ Generated {len(training_data)} examples (Total: {self.progress['executor']['generated']}/{self.config['executor_target']})")

                # Rate limiting
                time.sleep(3.0)

                # Check if we've reached target
                if self.progress["executor"]["generated"] >= self.config["executor_target"]:
                    break

            print(f"✅ Executor generation complete: {self.progress['executor']['generated']} examples")

        except Exception as e:
            print(f"❌ Executor generation failed: {e}")
            traceback.print_exc()

    def generate_critic_full_dataset(self, gsm8k_examples):
        """Generate complete Critic dataset"""
        print(f"\n🎯 Generating Critic Dataset (Target: {self.config['critic_target']} examples)")

        # Load checkpoint
        self.load_checkpoint("critic")

        if self.progress["critic"]["generated"] >= self.config["critic_target"]:
            print(f"✅ Critic dataset already complete: {self.progress['critic']['generated']} examples")
            return

        try:
            generator = CriticDataGenerator()
            remaining = self.config["critic_target"] - self.progress["critic"]["generated"]

            print(f"📝 Generating {remaining} remaining critic examples...")

            # Process in batches
            batch_size = self.config["batch_size"]
            examples_per_problem = 3  # Multiple error types per problem
            problems_per_batch = batch_size // examples_per_problem

            num_batches = math.ceil(len(gsm8k_examples) / problems_per_batch)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * problems_per_batch
                end_idx = min(start_idx + problems_per_batch, len(gsm8k_examples))
                batch_gsm8k = gsm8k_examples[start_idx:end_idx]

                print(f"  Batch {batch_idx + 1}/{num_batches}: Processing {len(batch_gsm8k)} GSM8K problems...")

                # Generate base examples
                target_examples = len(batch_gsm8k) * examples_per_problem
                base_examples = generator.generate_examples(batch_gsm8k, target_examples)

                if base_examples:
                    # Generate diverse error patterns
                    diverse_examples = generator.generate_diverse_error_examples(
                        base_examples, diversity_factor=2
                    )

                    training_data = generator.create_training_format(diverse_examples)

                    # Update progress
                    self.progress["critic"]["generated"] += len(training_data)

                    # Save checkpoint
                    append = batch_idx > 0 or self.progress["critic"]["generated"] > len(training_data)
                    self.save_checkpoint("critic", training_data, append=append)

                    print(f"    ✅ Generated {len(training_data)} examples (Total: {self.progress['critic']['generated']}/{self.config['critic_target']})")

                # Rate limiting
                time.sleep(3.0)

                # Check if we've reached target
                if self.progress["critic"]["generated"] >= self.config["critic_target"]:
                    break

            print(f"✅ Critic generation complete: {self.progress['critic']['generated']} examples")

        except Exception as e:
            print(f"❌ Critic generation failed: {e}")
            traceback.print_exc()

    def run_full_pipeline(self):
        """Run the complete full-scale pipeline"""
        print("🚀 FULL-SCALE ACOR SFT DATA GENERATION PIPELINE")
        print("=" * 80)
        print(f"🎯 Targets: Planner({self.config['planner_target']}) | Executor({self.config['executor_target']}) | Critic({self.config['critic_target']})")
        print("=" * 80)

        try:
            # Load GSM8K dataset
            print("📚 Loading full GSM8K training dataset...")
            gsm8k_loader = GSM8KLoader()
            raw_gsm8k = gsm8k_loader.load_dataset("train")

            if not raw_gsm8k:
                raise ValueError("Failed to load GSM8K dataset")

            # Use full dataset or specified size
            gsm8k_size = min(self.config["gsm8k_size"], len(raw_gsm8k))
            gsm8k_examples = gsm8k_loader.preprocess_for_training(raw_gsm8k[:gsm8k_size])

            print(f"✅ Loaded {len(gsm8k_examples)} GSM8K examples for processing")

            # Generate datasets for each module
            print(f"\n🏁 Starting full-scale generation...")

            # Planner
            self.generate_planner_full_dataset(gsm8k_examples)

            # Executor
            self.generate_executor_full_dataset(gsm8k_examples)

            # Critic
            self.generate_critic_full_dataset(gsm8k_examples)

            # Generate final summary
            total_generated = sum(self.progress[module]["generated"] for module in ["planner", "executor", "critic"])

            summary = {
                "pipeline_completed": datetime.now().isoformat(),
                "gsm8k_examples_used": len(gsm8k_examples),
                "config": self.config,
                "final_counts": {
                    "planner": self.progress["planner"]["generated"],
                    "executor": self.progress["executor"]["generated"],
                    "critic": self.progress["critic"]["generated"],
                    "total": total_generated
                },
                "output_files": {
                    "planner": str(self.output_dir / "planner_full_sft_dataset.jsonl"),
                    "executor": str(self.output_dir / "executor_full_sft_dataset.jsonl"),
                    "critic": str(self.output_dir / "critic_full_sft_dataset.jsonl")
                }
            }

            with open(self.output_dir / "full_pipeline_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            print("\n" + "=" * 80)
            print("🎉 FULL-SCALE PIPELINE COMPLETED!")
            print("=" * 80)
            print(f"📊 Final Counts:")
            print(f"   Planner: {self.progress['planner']['generated']:,} examples")
            print(f"   Executor: {self.progress['executor']['generated']:,} examples")
            print(f"   Critic: {self.progress['critic']['generated']:,} examples")
            print(f"   📈 Total: {total_generated:,} SFT examples generated")
            print(f"📂 Output directory: {self.output_dir}")

            return True

        except Exception as e:
            print(f"❌ Full pipeline failed: {e}")
            traceback.print_exc()
            return False

def main():
    """Main execution"""
    pipeline = FullScaleACORPipeline()

    print("🔥 Starting FULL-SCALE ACOR SFT Dataset Generation")
    print("⚠️  This will generate 50,000+ examples and may take 3-4 hours")
    print("💾 Progress is saved with checkpoints - you can resume if interrupted")
    print()

    # Run the pipeline
    success = pipeline.run_full_pipeline()

    if success:
        print("\n🎉 SUCCESS: Full-scale SFT datasets generated!")
        print("🚀 Ready for Llama 3.1 8B fine-tuning with DoRA adapters")
    else:
        print("\n❌ Pipeline failed. Check logs and resume from checkpoints.")

if __name__ == "__main__":
    main()