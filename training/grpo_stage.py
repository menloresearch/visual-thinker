#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO Training Script for LLM Fine-tuning
"""

import os
import re
import argparse
import logging
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

import torch
from datasets import load_dataset
from transformers import TextStreamer

from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Fine-tuning with GRPO')
    
    # Model configuration
    parser.add_argument('--model_name', type=str, default="jan-hq/Deepseek-Qwen2.5-7B-Redistil",
                        help='Pretrained model name')
    parser.add_argument('--max_seq_length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA rank parameter')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Load model in 4-bit quantization')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.4,
                        help='GPU memory utilization (0-1)')
                        
    # Training configuration
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Maximum number of training steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_generations', type=int, default=4,
                        help='Number of generations for GRPO')
    parser.add_argument('--max_prompt_length', type=int, default=612,
                        help='Maximum prompt length')
    parser.add_argument('--max_completion_length', type=int, default=4096,
                        help='Maximum completion length')
    parser.add_argument('--logging_steps', type=int, default=1,
                        help='Logging steps')
    parser.add_argument('--save_steps', type=int, default=200,
                        help='Save checkpoint steps')
                        
    # Dataset configuration
    parser.add_argument('--dataset_name', type=str, default="homebrewltd/Maze-Reasoning-filter",
                        help='HuggingFace dataset name')
    parser.add_argument('--dataset_split', type=str, default="train",
                        help='Dataset split to use')
                        
    # Output configuration
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saved models')
    parser.add_argument('--report_to', type=str, default="none",
                        help='Reporting integration (none, wandb, etc.)')
    
    return parser.parse_args()

def setup_model(args: argparse.Namespace) -> Tuple[Any, Any]:
    """
    Set up the FastLanguageModel with PEFT configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple containing the model and tokenizer
    """
    logger.info(f"Loading model: {args.model_name}")
    
    # Patch FastRL with GRPO
    PatchFastRL("GRPO", FastLanguageModel)
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Apply PEFT configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=42,
    )
    
    logger.info(f"Model loaded and configured with LoRA rank: {args.lora_rank}")
    return model, tokenizer
def load_training_data(args: argparse.Namespace) -> Dict:
    """
    Load and preprocess the training dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        Processed dataset
    """
    logger.info(f"Loading dataset: {args.dataset_name} (split: {args.dataset_split})")
    
    train_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # Preprocess dataset
    dataset = train_ds.map(lambda x: {
        'prompt': [
            {'role': 'user', 'content': x['Prompt']}
        ],
        'answer': x['Response'].strip()
    })
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    return dataset

def extract_xml_answer(text: str) -> str:
    """
    Extract answer from XML-formatted text.
    
    Args:
        text: Input text with XML tags
        
    Returns:
        Extracted answer text
    """
    try:
        answer = text.split("</think>")[1]
        return answer.strip()
    except:
        return ""

def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    """
    Reward function that checks correctness of answers.
    
    Args:
        prompts: Input prompts
        completions: Model completions
        answer: Ground truth answers
        
    Returns:
        List of reward scores
    """
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.debug('-'*20)
    logger.debug(f"Question:\n{q}")
    logger.debug(f"\nAnswer:\n{answer[0]}")
    logger.debug(f"\nResponse:\n{responses[0]}")
    logger.debug(f"\nExtracted:\n{extracted_responses[0]}")
    for r, a in zip(extracted_responses, answer):
        if r == a:
            direction = r.split("|><|")
            rewards.append(len(direction)*0.2)
        else:
            rewards.append(0.0)
    return rewards

def int_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward function that checks if responses contain valid direction tokens.
    
    Args:
        completions: Model completions
        
    Returns:
        List of reward scores
    """
    allowed_tokens = {"<|up|>", "<|down|>", "<|right|>", "<|left|>"}
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    def is_valid_sequence(seq):
        
        seq_no_whitespace = re.sub(r'\s+', '', seq)
        if not seq_no_whitespace:
            return False
        found_tokens = re.findall(r'<\|(?:up|down|right|left)\|>', seq_no_whitespace)
        reconstructed = ''.join(found_tokens)
        if reconstructed != seq_no_whitespace:
            return False
        return all(token in allowed_tokens for token in found_tokens)
    
    return [1.0 if is_valid_sequence(r) else 0.0 for r in extracted_responses]


# def strict_format_reward_func(completions, **kwargs) -> List[float]:
#     """
#     Reward function that checks if completions strictly follow the required format.
    
#     Args:
#         completions: Model completions
        
#     Returns:
#         List of reward scores
#     """
#     pattern = r"^<think>\n.*?\n</think>\n\n.*?\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r, re.DOTALL) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]


# def soft_format_reward_func(completions, **kwargs) -> List[float]:
#     """
#     Reward function that checks if completions loosely follow the required format.
    
#     Args:
#         completions: Model completions
        
#     Returns:
#         List of reward scores
#     """
#     pattern = r"<think>.*?</think>\s*.*?"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r, re.DOTALL) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]
def count_xml(text) -> float:
    """
    Count XML tags in response.
    
    Args:
        text: Input text
        
    Returns:
        Score based on XML tag presence
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    """
    Reward function based on proper XML tag usage.
    
    Args:
        completions: Model completions
        
    Returns:
        List of reward scores
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def create_training_args(args: argparse.Namespace) -> GRPOConfig:
    """
    Create GRPO training configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        GRPOConfig object
    """
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model_name.split('/')[-1]}"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to=args.report_to,
        output_dir=args.output_dir,
    )
    
    logger.info(f"Training configuration created. Output directory: {args.output_dir}")
    return training_args


def main():
    """Main execution function."""
    args = parse_arguments()

    model, tokenizer = setup_model(args)
    
    dataset = load_training_data(args)

    training_args = create_training_args(args)
    logger.info("Initializing GRPO Trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            # soft_format_reward_func,
            # strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

if __name__ == "__main__":
    main()