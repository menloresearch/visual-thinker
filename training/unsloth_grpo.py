from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
import torch
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import GRPOConfig, GRPOTrainer
import re

PatchFastRL("GRPO", FastLanguageModel)
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 128 # Larger rank = smarter, but slower
model_name = "jan-hq/Deepseek-Qwen2.5-7B-Redistil"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False, 
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 42,
)
train_ds = load_dataset("homebrewltd/Maze-Reasoning-filter", split="train")
def apply_template(examples):
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['Response'].strip()
    }) # type: ignore
    return data # type: ignore
dataset = train_ds.map(apply_template, batched=True)

def extract_xml_answer(text: str) -> str:
    answer = answer.split("</think>")[1]
    return answer.strip()
# define the reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n\n.*?\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*.*?"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# def count_xml(text) -> float:
#     count = 0.0
#     if text.count("<think>\n") == 1:
#         count += 0.125
#     if text.count("\n</think>\n") == 1:
#         count += 0.125
#     return count

# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]


training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 1e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 8192,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 10000,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = f"outputs/{model_name.split('/')[-1]}",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        # xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()