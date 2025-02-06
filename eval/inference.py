from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from datasets import load_dataset
model_path = "jan-hq/Deepseek-Qwen2.5-1.5B-Redistil-cp-500"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
test_ds = load_dataset("homebrewltd/Maze-Reasoning", split="test")
for instruction, label in zip(test_ds['Prompt'], test_ds['Response']):
    text = f"<|start_header_id|>user<|end_header_id|>\n\n<|reserved_special_token_69|>{instruction.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096,
    do_sample=False,
    temperature=0.6,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Output: {output}")
    print(f"Label: {label}")