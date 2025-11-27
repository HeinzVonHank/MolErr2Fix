import base64
import os
import time
from functools import wraps


class InterfaceCore:
    """Base interface for LLM API calls with rate limiting."""

    def __init__(self, model, api_key=None, api_name=None, rpm=-1):
        self.model_name = model
        self.rpm = rpm

        if api_key:
            self.api_key = api_key
        elif api_name:
            self.api_key = os.environ.get(api_name)
        else:
            self.api_key = None

        if not self.api_key:
            raise ValueError(f"API key for {self.model_name} not found")

    @classmethod
    def add_rpm_limit(cls, func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.rpm == -1:
                return func(self, *args, **kwargs)

            if not hasattr(self, '_call_times'):
                self._call_times = []

            current_time = time.time()
            self._call_times = [t for t in self._call_times if current_time - t < 60]

            if len(self._call_times) >= self.rpm:
                oldest_time = min(self._call_times)
                delay = 60 - (current_time - oldest_time)
                time.sleep(delay)

            self._call_times.append(current_time)
            return func(self, *args, **kwargs)
        return wrapper

    def inference_text_only(self, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1024):
        raise NotImplementedError


class GPT4Interface(InterfaceCore):
    def __init__(self, model="gpt-4", api_key=None, api_name="OPENAI_API_KEY", rpm=-1):
        import openai
        super().__init__(model=model, api_key=api_key, api_name=api_name, rpm=rpm)
        openai.api_key = self.api_key
        self.openai = openai

    @InterfaceCore.add_rpm_limit
    def inference_text_only(self, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1024):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
        response = self.openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class GeminiInterface(InterfaceCore):
    def __init__(self, model="gemini-2.0-flash", api_key=None, api_name="GEMINI_API_KEY", rpm=15):
        import google.generativeai as genai
        super().__init__(model=model, api_key=api_key, api_name=api_name, rpm=rpm)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    @InterfaceCore.add_rpm_limit
    def inference_text_only(self, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1000):
        response = self.model.generate_content(
            contents=[{"role": "user", "parts": [system_message + "\n\n" + query]}],
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        return response.text.strip()


class ClaudeInterface(InterfaceCore):
    def __init__(self, model="claude-3-7-sonnet-20250219", api_key=None, api_name="ANTHROPIC_API_KEY", rpm=-1):
        import anthropic
        super().__init__(model=model, api_key=api_key, api_name=api_name, rpm=rpm)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @InterfaceCore.add_rpm_limit
    def inference_text_only(self, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1024):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": system_message + "\n\n" + query}]
        )
        return response.content[0].text.strip()

    @InterfaceCore.add_rpm_limit
    def inference_with_image(self, image_path, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1024):
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                    {"type": "text", "text": system_message + "\n\n" + query}
                ]
            }]
        )
        return response.content[0].text.strip()


class GrokInterface(InterfaceCore):
    def __init__(self, model="grok-3", api_key=None, api_name="XAI_API_KEY", rpm=600):
        from openai import OpenAI
        super().__init__(model=model, api_key=api_key, api_name=api_name, rpm=rpm)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

    @InterfaceCore.add_rpm_limit
    def inference_text_only(self, query, system_message="You are a helpful assistant.", temperature=0.7, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
        )
        return response.choices[0].message.content.strip()


def get_llm_interface(model_name, api_key=None):
    """Factory function to get appropriate LLM interface."""
    if "gemini" in model_name:
        return GeminiInterface(model=model_name, api_key=api_key)
    elif "claude" in model_name:
        return ClaudeInterface(model=model_name, api_key=api_key)
    elif "grok" in model_name:
        return GrokInterface(model=model_name, api_key=api_key)
    else:
        return GPT4Interface(model=model_name, api_key=api_key)
