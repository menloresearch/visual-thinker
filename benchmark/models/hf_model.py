from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base import BaseModel
from typing import List

class HuggingFaceModel(BaseModel):
    def __init__(self, 
                model_name: str,
                prompt_template: str,
                batch_size: int = 8,
                max_tokens: int = 512,
                temperature: float = 0.0):
        super().__init__(model_name, batch_size)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.prompt_template = prompt_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
    def generate(self, prompts: List[str]) -> List[str]:
        format_prompts = []
        for prompt in prompts:
            chat = [
                {"role": "user", "content": prompt.strip()}
            ]
            format_prompts.append(chat)
        inputs = self.tokenizer.apply_chat_template(format_prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.temperature > 0
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)