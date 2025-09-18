import time, yaml, base64, os
from typing import List, Dict, Tuple
from functools import wraps

class InterfaceCore:
    def __init__(self, model: str, api_key: str = None, API_name: str = None, RPM=-1):
        self.model_name = model
        self.RPM = RPM
        
        if api_key:
            self.api_key = api_key
        elif API_name:
            self.api_key = os.environ.get(API_name)
        else:
            self.api_key = None

        if not self.api_key:
            raise ValueError(
                f"API key for {self.model_name} is not provided. "
                "Please set it as an environment variable or pass it directly."
            )

    @classmethod
    def add_RPM_limit(cls, func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.RPM == -1:
                return func(self, *args, **kwargs)

            if not hasattr(self, '_call_times'):
                self._call_times = []
            
            current_time = time.time()
            
            self._call_times = [t for t in self._call_times if current_time - t < 60]
            
            if len(self._call_times) >= self.RPM:
                oldest_time = min(self._call_times)
                time_since_oldest = current_time - oldest_time
                if time_since_oldest < 60:
                    delay = 60 - time_since_oldest
                    print(f"Delaying for {delay:.2f} seconds to respect RPM limit.")
                    time.sleep(delay)
            
            self._call_times.append(current_time)
            
            return func(self, *args, **kwargs)
        return wrapper
    
    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

# ==============================================================================

import openai

class GPT4Interface(InterfaceCore):
    def __init__(self, model="gpt-4", api_key=None, API_name="OPENAI_API_KEY", RPM=-1):
        super().__init__(model=model, api_key=api_key, API_name=API_name, RPM=RPM)
        openai.api_key = self.api_key

    @InterfaceCore.add_RPM_limit
    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
        
# ==============================================================================

import google.generativeai as genai

class GeminiInterface(InterfaceCore):
    def __init__(self, model="gemini-2.0-flash", api_key=None, API_name="GEMINI_API_KEY", RPM=15):
        super().__init__(model=model, api_key=api_key, API_name=API_name, RPM=RPM)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    @InterfaceCore.add_RPM_limit
    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            response = self.model.generate_content(
                contents=[
                    {"role": "user", "parts": [system_message + "\n\n" + query]}
                ],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ==============================================================================

import anthropic

class ClaudeInterface(InterfaceCore):
    def __init__(self, model="claude-3-7-sonnet-20250219", api_key=None, API_name="ANTHROPIC_API_KEY", RPM=-1):
        super().__init__(model=model, api_key=api_key, API_name=API_name, RPM=RPM)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @InterfaceCore.add_RPM_limit
    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": system_message + "\n\n" + query}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    @InterfaceCore.add_RPM_limit
    def inference_with_image(self, image_path: str, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            with open(image_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": system_message + "\n\n" + query
                            }
                        ]
                    }
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"
        
# ==============================================================================

from openai import OpenAI

class GrokInterface(InterfaceCore):
    def __init__(self, model="grok-3", api_key=None, API_name="XAI_API_KEY", RPM=600):
        super().__init__(model=model, api_key=api_key, API_name=API_name, RPM=RPM)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1",)

    @InterfaceCore.add_RPM_limit
    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ==============================================================================

def get_llm_interfance(model_name: str, api_key: str = None):
    if "gemini" in model_name:
        llm_api = GeminiInterface(model=model_name, api_key=api_key)
    elif "claude" in model_name:
        llm_api = ClaudeInterface(model=model_name, api_key=api_key)
    elif "grok" in model_name:
        llm_api = GrokInterface(model=model_name, api_key=api_key)
    else:
        llm_api = GPT4Interface(model=model_name, api_key=api_key)
    return llm_api