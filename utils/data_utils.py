"""
Data utilities for ACOR system
"""
import json
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizer


def format_training_data(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> List[Dict[str, Any]]:
    """
    Format training examples for instruction tuning

    Args:
        examples: List of training examples with instruction/input/output
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        List of formatted examples with input_ids, attention_mask, labels
    """
    formatted_examples = []

    for example in examples:
        # Create instruction-following format
        if 'instruction' in example and 'input' in example and 'output' in example:
            # Standard instruction format
            prompt = f"### Instruction:\n{example['instruction']}\n\n"
            if example['input'].strip():
                prompt += f"### Input:\n{example['input']}\n\n"
            prompt += "### Response:\n"
            response = example['output']

        elif 'question' in example and 'answer' in example:
            # Simple Q&A format
            prompt = f"Question: {example['question']}\nAnswer: "
            response = example['answer']

        else:
            # Skip malformed examples
            continue

        # Combine prompt and response for training
        full_text = prompt + response

        # Tokenize
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

        # Create labels (same as input_ids)
        labels = tokenized['input_ids'].copy()

        # Calculate prompt length to mask it in loss calculation
        prompt_tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

        prompt_length = len(prompt_tokenized['input_ids'])

        # Mask prompt tokens in labels (set to -100)
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100

        formatted_example = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }

        formatted_examples.append(formatted_example)

    return formatted_examples


def create_conversation_format(
    instruction: str,
    input_text: str = "",
    output_text: str = ""
) -> str:
    """
    Create a conversation format for training

    Args:
        instruction: Task instruction
        input_text: Input content
        output_text: Expected output

    Returns:
        Formatted conversation string
    """
    conversation = f"### Instruction:\n{instruction}\n\n"

    if input_text.strip():
        conversation += f"### Input:\n{input_text}\n\n"

    conversation += "### Response:\n"

    if output_text:
        conversation += output_text

    return conversation


def validate_training_example(example: Dict[str, Any]) -> bool:
    """
    Validate a training example

    Args:
        example: Training example to validate

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if 'instruction' in example:
        required_fields = ['instruction', 'input', 'output']
    elif 'question' in example:
        required_fields = ['question', 'answer']
    else:
        return False

    for field in required_fields:
        if field not in example or not isinstance(example[field], str):
            return False

    # Check content length
    if 'output' in example and len(example['output']) < 5:
        return False

    if 'answer' in example and len(example['answer']) < 5:
        return False

    return True


def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True
) -> tuple:
    """
    Split dataset into train/validation/test sets

    Args:
        data: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    import random

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    data_copy = data.copy()

    if shuffle:
        random.shuffle(data_copy)

    total_size = len(data_copy)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]

    return train_data, val_data, test_data


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')