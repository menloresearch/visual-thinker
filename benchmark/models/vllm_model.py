from vllm import LLM, SamplingParams
from .base import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer

class VLLMModel(BaseModel):
    def __init__(self, 
                model_name: str,
                prompt_template: str,
                batch_size: int = 8,
                max_tokens: int = 512,
                temperature: float = 0.0,
                tensor_parallel_size: int = 1):
        super().__init__(model_name, batch_size)
        
        self.llm = LLM(
            model=model_name,
            # tensor_parallel_size=tensor_parallel_size,
            # trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prompt_template = prompt_template
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=1.0,
            # top_k=-1
        )

    def generate(self, prompts: List[str]) -> List[str]:
        format_prompts = []
        for prompt in prompts:
            chat = [
                {"role": "user", "content": prompt.strip()}
            ]
            format_prompts.append(chat)
        inputs = self.tokenizer.apply_chat_template(format_prompts, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        outputs = self.llm.generate(inputs, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]