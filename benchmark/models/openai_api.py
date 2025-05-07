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
                 prompt_template: str,
                 batch_size: int = 8,
                 api_key: Optional[str] = None,
                 api_base: str = "https://api.openai.com/v1",
                 max_tokens: int = 512,
                 temperature: float = 0.0,
                 request_timeout: int = 60,
                 retry_attempts: int = 3,
                 retry_delay: int = 5,
                 organization: Optional[str] = None,
                 skip_auth: bool = False,
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
            skip_auth (bool): Skip authentication for local endpoints
            **kwargs: Additional parameters
        """
        super().__init__(model_name, prompt_template, batch_size)
        
        self.api_key = api_key or self._get_api_key_from_env()
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.organization = organization
        self.skip_auth = skip_auth
        
        # Only require API key for non-local endpoints unless skip_auth is True
        if not self.api_key and not self.skip_auth:
            raise ValueError(
                "API key is required for non-local endpoints. Provide it either as a parameter or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Only add authorization for non-local endpoints or if explicitly provided
        if self.api_key and not self.skip_auth:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        if self.organization and not self.skip_auth:
            self.headers["OpenAI-Organization"] = self.organization
            
        # Determine API endpoint
        self.chat_endpoint = f"{self.api_base}/chat/completions"
        
        logging.info(f"Initialized OpenAI-compatible model: {model_name}")
        logging.info(f"API endpoint: {self.chat_endpoint}")
            
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
        
        payload = {
            "model": self.model_name,
            "messages": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            logging.debug(f"Making API request to {self.chat_endpoint}")
            response = requests.post(
                self.chat_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                logging.error(error_msg)
                raise Exception(error_msg)
            
            # Parse response
            response_data = response.json()
            logging.debug(f"API response: {json.dumps(response_data)[:200]}...")
            
            generated_text = ""
            
            # Standard OpenAI format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    generated_text = choice["message"]["content"]
                elif "text" in choice:  # Some APIs use 'text' directly
                    generated_text = choice["text"]
            
            # Alternative format used by some providers
            elif "response" in response_data:
                generated_text = response_data["response"]
            
            # If we still don't have text, log a warning but return what we have
            if not generated_text:
                logging.warning(f"Could not extract generated text from response: {response_data}")
                return str(response_data)
            
            return f"</think>{generated_text}"
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error when contacting API: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from API: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

