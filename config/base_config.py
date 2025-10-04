"""
Base configuration for ACOR system
"""
from dataclasses import dataclass
from typing import List, Optional
import os
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_length: int = 2048
    temperature: float = 0.1
    do_sample: bool = True
    top_p: float = 0.9
    pad_token_id: Optional[int] = None

@dataclass
class DoRAConfig:
    """Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation)"""
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.1
    target_modules: List[str] = None
    modules_to_save: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        if self.modules_to_save is None:
            self.modules_to_save = ["embed_tokens", "lm_head"]

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10
    fp16: bool = True

@dataclass
class DataConfig:
    """Configuration for data processing"""
    gsm8k_train_size: int = 7500
    synthetic_data_size: int = 20000
    augmentation_factor: int = 3
    validation_split: float = 0.1
    max_input_length: int = 1024
    max_output_length: int = 512

@dataclass
class ACORConfig:
    """Main ACOR system configuration"""
    max_correction_attempts: int = 3
    correction_threshold: float = 0.7
    enable_logging: bool = True
    log_level: str = "INFO"

    # Model configurations
    model: ModelConfig = None
    dora: DoRAConfig = None
    training: TrainingConfig = None
    data: DataConfig = None

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = None
    models_dir: Path = None
    logs_dir: Path = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.dora is None:
            self.dora = DoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

        # Set up paths
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "training" / "checkpoints"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

# Environment variables
LLM_GW_EXPRESS_KEY = os.getenv("LLM_GW_EXPRESS_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "acor-system")
HF_TOKEN = os.getenv("HF_TOKEN")