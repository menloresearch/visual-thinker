from unsloth import FastLanguageModel
max_seq_length=4096
dtype = "bfloat16"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./outputs/Deepseek-Qwen2.5-7B-Redistil/checkpoint-400",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False,
)
model.push_to_hub_merged("jan-hq/Deepseek-Qwen2.5-7B-Redistil-GRPO", tokenizer, save_method = "merged_16bit")
model.push_to_hub_gguf(
    "jan-hq/Deepseek-Qwen2.5-7B-Redistil-GRPO", # Change hf to your username!
    tokenizer,
    quantization_method = ["q4_k_m"],
)