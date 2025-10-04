"""
Base module class for ACOR components
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from config.base_config import ACORConfig, ModelConfig


class BaseACORModule(ABC):
    """Base class for all ACOR modules (Planner, Executor, Critic)"""

    def __init__(
        self,
        module_name: str,
        config: ACORConfig,
        adapter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.module_name = module_name
        self.config = config
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set up logging
        self.logger = logging.getLogger(f"acor.{module_name}")

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.generation_config = None

        self._load_model()
        self._setup_generation_config()

    def _load_model(self):
        """Load the base model and tokenizer"""
        self.logger.info(f"Loading base model: {self.config.model.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            trust_remote_code=True
        )

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.config.model.pad_token_id = self.tokenizer.pad_token_id

        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # Load adapter if provided
        if self.adapter_path and Path(self.adapter_path).exists():
            self.logger.info(f"Loading adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False
            )

        self.model.eval()

    def _setup_generation_config(self):
        """Set up generation configuration"""
        self.generation_config = GenerationConfig(
            max_length=self.config.model.max_length,
            temperature=self.config.model.temperature,
            do_sample=self.config.model.do_sample,
            top_p=self.config.model.top_p,
            pad_token_id=self.config.model.pad_token_id or self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0
        )

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate response from the model"""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_length - max_new_tokens
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        return response

    @abstractmethod
    def create_prompt(self, **kwargs) -> str:
        """Create module-specific prompt"""
        pass

    @abstractmethod
    def process_input(self, **kwargs) -> str:
        """Process input and generate response"""
        pass

    @abstractmethod
    def parse_output(self, output: str) -> Dict:
        """Parse and structure the model output"""
        pass