from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base import BaseModel
from typing import List

class HuggingFaceModel(BaseModel):
    def __init__(self, 
                 model_name: str,
                 batch_size: int = 8,
                 max_tokens: int = 512,
                 temperature: float = 0.0):
        super().__init__(model_name, batch_size)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.temperature > 0
        )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)