"""
Base data generator for ACOR system
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path

import subprocess
import pandas as pd
from tqdm import tqdm

from config.base_config import LLM_GW_EXPRESS_KEY


class BaseDataGenerator(ABC):
    """Base class for all data generators in ACOR system"""

    def __init__(
        self,
        teacher_model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3
    ):
        self.teacher_model = teacher_model
        self.api_key = api_key or LLM_GW_EXPRESS_KEY
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.gateway_url = "https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl/chat/completions"

        # Set up logging
        self.logger = logging.getLogger(f"acor.data_generation.{self.__class__.__name__}")

        # Validate API key
        if not self.api_key:
            raise ValueError("LLM Gateway API key is required for data generation")

        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0
        }

    def call_teacher_model(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> Optional[str]:
        """
        Make a call to the teacher model via internal gateway with retry logic

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated text or None if failed
        """
        self.stats['total_requests'] += 1

        for attempt in range(self.max_retries):
            try:
                # Prepare the request payload
                payload = {
                    "model": self.teacher_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9
                }

                # Create curl command
                curl_cmd = [
                    "curl", "-X", "POST", self.gateway_url,
                    "--header", f"Authorization: Bearer {self.api_key}",
                    "--header", "Content-Type: application/json",
                    "--data", json.dumps(payload),
                    "--silent"
                ]

                # Execute curl command
                result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=60)

                if result.returncode != 0:
                    raise Exception(f"Curl command failed with return code {result.returncode}: {result.stderr}")

                # Parse response
                response_data = json.loads(result.stdout)

                # Check for errors in response
                if "error" in response_data:
                    error_msg = response_data["error"].get("message", "Unknown API error")
                    if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        raise Exception(f"Rate limit: {error_msg}")
                    else:
                        raise Exception(f"API error: {error_msg}")

                # Extract content - handle different response formats
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    choice = response_data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    elif "text" in choice:  # Alternative format
                        content = choice["text"]
                    else:
                        raise Exception(f"Unexpected response format: {choice}")
                else:
                    raise Exception(f"No choices in response: {response_data}")
                self.stats['successful_requests'] += 1

                # Track token usage if available
                if "usage" in response_data:
                    self.stats['total_tokens_used'] += response_data["usage"].get("total_tokens", 0)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return content

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse API response on attempt {attempt + 1}: {e}")
                self.logger.error(f"Response: {result.stdout if 'result' in locals() else 'N/A'}")
                if attempt == self.max_retries - 1:
                    break
                time.sleep(1)

            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "limit" in error_msg:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        break
                    time.sleep(1)

        self.stats['failed_requests'] += 1
        return None

    def save_data(
        self,
        data: List[Dict[str, Any]],
        output_path: Path,
        format_type: str = "jsonl"
    ) -> bool:
        """
        Save generated data to file

        Args:
            data: List of data examples
            output_path: Path to save the data
            format_type: Format to save in ('jsonl', 'json', 'csv')

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

            elif format_type == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            elif format_type == "csv":
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False, encoding='utf-8')

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            self.logger.info(f"Saved {len(data)} examples to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save data to {output_path}: {e}")
            return False

    def load_data(self, input_path: Path, format_type: str = "jsonl") -> List[Dict[str, Any]]:
        """
        Load data from file

        Args:
            input_path: Path to load data from
            format_type: Format to load from ('jsonl', 'json', 'csv')

        Returns:
            List of data examples
        """
        try:
            if format_type == "jsonl":
                data = []
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data

            elif format_type == "json":
                with open(input_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

            elif format_type == "csv":
                df = pd.read_csv(input_path, encoding='utf-8')
                return df.to_dict('records')

            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            self.logger.error(f"Failed to load data from {input_path}: {e}")
            return []

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate a single training example

        Args:
            example: Training example to validate

        Returns:
            True if valid, False otherwise
        """
        return self._validate_example_impl(example)

    @abstractmethod
    def _validate_example_impl(self, example: Dict[str, Any]) -> bool:
        """Implementation-specific validation logic"""
        pass

    @abstractmethod
    def generate_examples(
        self,
        input_data: List[Dict[str, Any]],
        num_examples: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples

        Args:
            input_data: Source data to process
            num_examples: Number of examples to generate
            **kwargs: Additional arguments

        Returns:
            List of generated training examples
        """
        pass

    def generate_and_save(
        self,
        input_data: List[Dict[str, Any]],
        output_path: Path,
        num_examples: int,
        batch_size: int = 50,
        format_type: str = "jsonl",
        **kwargs
    ) -> bool:
        """
        Generate examples and save them to file

        Args:
            input_data: Source data
            output_path: Path to save generated data
            num_examples: Number of examples to generate
            batch_size: Process in batches of this size
            format_type: Format to save in
            **kwargs: Additional generation arguments

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Generating {num_examples} examples, processing in batches of {batch_size}")

        all_examples = []
        batches = [input_data[i:i + batch_size] for i in range(0, len(input_data), batch_size)]

        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            batch_examples_needed = min(batch_size, num_examples - len(all_examples))
            if batch_examples_needed <= 0:
                break

            batch_examples = self.generate_examples(
                batch,
                batch_examples_needed,
                **kwargs
            )

            # Validate examples
            valid_examples = [ex for ex in batch_examples if self.validate_example(ex)]

            all_examples.extend(valid_examples)

            self.logger.info(
                f"Batch {batch_idx + 1}/{len(batches)}: "
                f"Generated {len(valid_examples)}/{len(batch_examples)} valid examples. "
                f"Total: {len(all_examples)}/{num_examples}"
            )

            if len(all_examples) >= num_examples:
                break

        # Trim to exact number requested
        final_examples = all_examples[:num_examples]

        # Save the data
        success = self.save_data(final_examples, output_path, format_type)

        if success:
            self.logger.info(f"Successfully generated and saved {len(final_examples)} examples")
            self._log_statistics()
        else:
            self.logger.error("Failed to save generated examples")

        return success

    def _log_statistics(self):
        """Log generation statistics"""
        self.logger.info("Generation Statistics:")
        self.logger.info(f"  Total API requests: {self.stats['total_requests']}")
        self.logger.info(f"  Successful requests: {self.stats['successful_requests']}")
        self.logger.info(f"  Failed requests: {self.stats['failed_requests']}")
        self.logger.info(f"  Success rate: {self.stats['successful_requests'] / max(self.stats['total_requests'], 1):.2%}")
        self.logger.info(f"  Total tokens used: {self.stats['total_tokens_used']}")

    def estimate_cost(self, num_examples: int, avg_tokens_per_example: int = 500) -> float:
        """
        Estimate the cost of generating examples

        Args:
            num_examples: Number of examples to generate
            avg_tokens_per_example: Average tokens per example

        Returns:
            Estimated cost in USD (using internal gateway - costs may vary)
        """
        # Claude Sonnet 4 pricing via internal gateway (estimated)
        # Note: Actual costs depend on internal gateway pricing
        input_cost_per_token = 0.003 / 1000  # Estimated internal rate
        output_cost_per_token = 0.015 / 1000  # Estimated internal rate

        # Rough estimation: 60% input, 40% output
        total_tokens = num_examples * avg_tokens_per_example
        input_tokens = total_tokens * 0.6
        output_tokens = total_tokens * 0.4

        estimated_cost = (input_tokens * input_cost_per_token +
                         output_tokens * output_cost_per_token)

        self.logger.info(f"Estimated cost for {num_examples} examples: ${estimated_cost:.2f} (internal gateway rates)")
        return estimated_cost