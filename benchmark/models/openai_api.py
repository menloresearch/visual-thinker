import requests
import json
import time
import logging
from typing import List, Optional, Dict, Any
from .base import BaseModel

class OpenAIModel(BaseModel):
    """
    Model implementation for OpenAI and OpenAI-compatible APIs (e.g., Azure, Claude, etc.)
    """
    
    def __init__(self, 
                 model_name: str,
                 batch_size: int = 8,
                 api_key: Optional[str] = None,
                 api_base: str = "https://api.openai.com/v1",
                 max_tokens: int = 512,
                 temperature: float = 0.0,
                 request_timeout: int = 60,
                 retry_attempts: int = 3,
                 retry_delay: int = 5,
                 organization: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI-compatible model.
        
        Args:
            model_name (str): Name of the model to use
            batch_size (int): Batch size for inference
            api_key (str, optional): API key for authentication
            api_base (str): Base URL for API requests
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            request_timeout (int): Timeout for API requests in seconds
            retry_attempts (int): Number of retry attempts for failed requests
            retry_delay (int): Delay between retries in seconds
            organization (str, optional): Organization ID for API requests
            **kwargs: Additional parameters
        """
        super().__init__(model_name, batch_size)
        
        self.api_key = api_key or self._get_api_key_from_env()
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.organization = organization
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it either as a parameter or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization:
            self.headers["OpenAI-Organization"] = self.organization
            
        # Determine API endpoint
        self.chat_endpoint = f"{self.api_base}/chat/completions"
            
        logging.info(f"Initialized OpenAI-compatible model: {model_name}")
        logging.info(f"API base: {api_base}")

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        return os.environ.get("OPENAI_API_KEY")

    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts (List[str]): List of prompts
            
        Returns:
            List[str]: List of generated responses
        """
        format_prompts = []
        for prompt in prompts:
            chat = [
                {"role": "user", "content": prompt.strip()}
            ]
            format_prompts.append(chat)
        results = []
        
        # Process prompts in the specified batch size
        for i in range(0, len(format_prompts), self.batch_size):
            batch = format_prompts[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
        return results

    def _process_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts.
        
        Args:
            prompts (List[str]): List of prompts
            
        Returns:
            List[str]: List of generated responses
        """
        batch_results = []
        
        # Make individual API requests for each prompt
        for prompt in prompts:
            for attempt in range(self.retry_attempts):
                try:
                    response = self._make_api_request(prompt)
                    batch_results.append(response)
                    break
                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        logging.warning(f"API request failed: {e}. Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        logging.error(f"API request failed after {self.retry_attempts} attempts: {e}")
                        batch_results.append("")  # Empty string for failed requests
        
        return batch_results

    def _make_api_request(self, prompt: str) -> str:
        """
        Make a single API request.
        
        Args:
            prompt (str): Prompt to send to the API
            
        Returns:
            str: Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            self.chat_endpoint,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=self.request_timeout
        )
        
        # Handle API errors
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}: {response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        # Parse response
        response_data = response.json()
        
        # Extract generated text
        generated_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Post-process to extract just the solution (assumes the model follows the expected format)
        solution = self._extract_solution(generated_text)
        
        return solution

    def _extract_solution(self, text: str) -> str:
        """
        Extract the solution from the generated text.
        Cleans up the response to extract only the sequence of moves.
        
        Args:
            text (str): Raw generated text
            
        Returns:
            str: Extracted solution
        """
        # Find sequences like <|up|>, <|down|>, etc.
        import re
        move_pattern = re.compile(r'<\|(up|down|left|right)\|>')
        moves = move_pattern.findall(text)
        
        # If no moves found, return the original text
        if not moves:
            return text
        
        # Reconstruct the proper format
        solution = " ".join([f"<|{move}|>" for move in moves])
        return solution