from abc import ABC, abstractmethod
from typing import List
from .instruction_type import prompt_templates

class BaseModel(ABC):
    """Base class for all model implementations."""
    
    @abstractmethod
    def __init__(self, model_name: str, prompt_template: str, batch_size: int = 8, **kwargs):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.model_config = kwargs

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass

    def format_prompt(self, maze_prompt: str) -> str:
        """Format the prompt for the model."""
        return prompt_templates[self.prompt_template].format(maze_prompt=maze_prompt)