from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from datasets import load_dataset
model_path = "jan-hq/Deepseek-Qwen2.5-1.5B-Redistil-cp-500"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
test_ds = load_dataset("homebrewltd/Maze-Reasoning", split="test")
for instruction, label in zip(test_ds['Prompt'], test_ds['Response']):
    chat = [
        {"role": "user", "content": instruction.strip()}
    ]
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
    tokenized_chat,
    max_new_tokens=20000,
    do_sample=True,
    temperature=0.6,
    )

    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Output: {output}")
    print(f"Label: {label}")