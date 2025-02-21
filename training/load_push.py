from unsloth import FastLanguageModel
max_seq_length=4096
dtype = "bfloat16"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./outputs/AlphaMaze-v0.2-1.5B/",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False,
)
model.push_to_hub_merged("jan-hq/AlphaMaze-v0.2-1.5B", tokenizer, save_method = "merged_16bit")
model.push_to_hub_gguf(
    "jan-hq/AlphaMaze-v0.2-1.5B", # Change hf to your username!
    tokenizer,
    quantization_method = ["q4_k_m"],
)