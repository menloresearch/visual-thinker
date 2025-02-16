from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    """Base class for all model implementations."""
    
    @abstractmethod
    def __init__(self, model_name: str, batch_size: int = 8, **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_config = kwargs

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        pass

    def format_prompt(self, maze_prompt: str) -> str:
        """Format the prompt for the model."""
        return f"""You are an AI maze solver. Given the maze below, provide a sequence of moves to reach the target.
Use only the following format for moves: <|up|>, <|down|>, <|left|>, <|right|>
Separate moves with spaces. Only output the sequence of moves, nothing else.

Maze:
{maze_prompt}

Solution:"""