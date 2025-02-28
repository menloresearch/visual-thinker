from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder
import torch
import os

device = torch.device("cpu")
# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.bfloat16) 
tokenizer = AutoTokenizer.from_pretrained(model_name)
old_vocab_size = len(tokenizer)
print(old_vocab_size)
coordinate_tokens = [f"<|{row}-{col}|>" for row in range(5) for col in range(5)] 
movement_tokens = ["<|up|>", "<|down|>", "<|left|>","<|right|>","<|blank|>", "<|origin|>","<|target|>"]
wall_token = [
    "<|no_wall|>",
    "<|up_wall|>",
    "<|down_wall|>",
    "<|left_wall|>",
    "<|right_wall|>",
    "<|up_down_wall|>",
    "<|up_left_wall|>",
    "<|up_right_wall|>",
    "<|down_left_wall|>",
    "<|down_right_wall|>",
    "<|left_right_wall|>",
    "<|up_down_left_wall|>",
    "<|up_down_right_wall|>",
    "<|up_left_right_wall|>",
    "<|down_left_right_wall|>",
    "<|all_wall|>",
]
add_tokens = coordinate_tokens + movement_tokens + wall_token
# open vocabulary for sound tokens and add special tokens
tokenizer.add_tokens(add_tokens)
# resize
total_new_tokens = 271 #fixme: cause number of added tokens is lower than number of padding(271)

print("--- Initializing new embedding with average weight ---")
print("___________________________________")
# Extract model parametersad 
params = model.state_dict()
# Get current embeddings
embeddings = params['model.embed_tokens.weight']
# create a new embedding layer with shape (old_vocab_size + total_new_tokens, embedding_dim) with the new embeddings initialized to 0
pre_expansion_embeddings = embeddings[:-total_new_tokens, :]
print(f"Pre-expansion embeddings shape: {pre_expansion_embeddings.shape}")

# Calculate mean and covariance
mu = torch.mean(pre_expansion_embeddings, dim=0).to(torch.float32).to(device)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
sigma = sigma.to(torch.float32).to(device)

# Sample new embeddings from multivariate normal distribution
dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
new_embeddings = torch.stack(tuple(dist.sample().to(device) for _ in range(total_new_tokens)), dim=0)
print(f"New embeddings shape: {new_embeddings.shape}")

# Update embeddings with new values
params['model.embed_tokens.weight'][-total_new_tokens:,:] = new_embeddings
print(params['model.embed_tokens.weight'][:-total_new_tokens, :])
print(f"Updated embedding weights shape: {params['model.embed_tokens.weight'].shape}")
print("___________________________________")


print("--- Initializing new lm_head with average weight ---")
print("___________________________________")

print(f"Updated embedding weights shape: {params['lm_head.weight'].shape}")
# Get current embeddings
embeddings = params['lm_head.weight']
pre_expansion_embeddings = embeddings[:-total_new_tokens, :]
print(f"Pre-expansion embeddings shape: {pre_expansion_embeddings.shape}")

# Calculate mean and covariance
mu = torch.mean(pre_expansion_embeddings, dim=0).to(torch.float32).to(device)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
sigma = sigma.to(torch.float32).to(device)

# Sample new embeddings from multivariate normal distribution
dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)
new_embeddings = torch.stack(tuple(dist.sample().to(device) for _ in range(total_new_tokens)), dim=0)
print(f"New embeddings shape: {new_embeddings.shape}")

# Update embeddings with new values
params['model.embed_tokens.weight'][-total_new_tokens:,:] = new_embeddings
print(f"Updated lm_head weights shape: {params['lm_head.weight'].shape}")
print("___________________________________")

# Load updated parameters into model
model.load_state_dict(params)
# # Save the updated model and tokenizer locally
output_dir = "AlphaMaze-1.5B-init/"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16) 
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model.push_to_hub("jan-hq/AlphaMaze-1.5B-init")
tokenizer.push_to_hub("jan-hq/AlphaMaze-1.5B-init")
print("Model and tokenizer updated and pushed to Hugging Face Hub.")