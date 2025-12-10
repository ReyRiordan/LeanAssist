import numpy as np
from typing import List, Tuple
import os
import requests
from openai import OpenAI
from .external_parser import Generator, Transformer, pre_process_input, post_process_output, choices_dedup


class UnifiedAPIRunner(Generator, Transformer):
    """Unified runner for both OpenRouter and Fireworks API"""

    def __init__(self, provider: str, model: str, temperature: float = 1.0, num_samples: int = 10, 
                 reasoning_enabled: bool = False, timeout: int = 45):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.num_samples = num_samples
        self.reasoning_enabled = reasoning_enabled
        self.timeout = timeout

        # Provider-specific clients
        if self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("missing OPENROUTER_API_KEY")
        elif self.provider == "fireworks":
            self.api_key = os.getenv("FIREWORKS_API_KEY")
            if not self.api_key:
                raise ValueError("missing FIREWORKS_API_KEY")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.fireworks.ai/inference/v1/"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def extract_tactic(self, response: str) -> str:
        """Extract tactic (copied over benchmarking code)"""
        text = response.strip()
        # ```lean
        if "```lean" in text:
            start = text.find("```lean") + len("```lean")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        # `
        elif '`' in text:
            start = text.find("`")
            end = text.find("`", start + 1)
            if end != -1:
                return text[start+1:end].strip()
        # raw
        return text

    def call_openrouter(self, messages: list) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            # "max_tokens": self.max_tokens,
            "reasoning":{
                "enabled": self.reasoning_enabled
            },
            "messages": messages
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        raw = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response = raw.json()

        if 'choices' not in response:
            raise ValueError(f"Unexpected API response: {response}")

        return response['choices'][0]['message']['content']

    def call_fireworks(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model, 
            temperature=self.temperature,
            # max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_enabled, # toggle reasoning
            messages=messages
        )
        
        return response.choices[0].message.content

    def generate(self, input: str, target_prefix: str = "") -> List[Tuple[str, float]]:
        """Generate tactics from proof state."""
        prompt = pre_process_input(self.model, input + target_prefix)
        messages = [{"role": "user", "content": prompt}]

        suggestions = []
        for i in range(self.num_samples):
            try:
                # Call appropriate API
                if self.provider == "openrouter":
                    response = self.call_openrouter(messages)
                elif self.provider == "fireworks":
                    response = self.call_fireworks(messages)

                tactic = self.extract_tactic(response)
                processed_tactic = post_process_output(self.model, tactic)
                if processed_tactic:
                    # Score based on generation order
                    score = float(self.num_samples - i)
                    suggestions.append((processed_tactic, score))

            except Exception as e:
                print(f"API call failed: {e}")
                continue

        # Use choices_dedup to handle duplicates and sort by score
        return choices_dedup(suggestions)


if __name__ == "__main__":
    # Test openrouter
    try:
        or_runner = UnifiedAPIRunner(
            provider="openrouter",
            model="anthropic/claude-haiku-4.5",
            num_samples=3,
            reasoning_enabled=False
        )
        result = or_runner.generate("n : ℕ\n⊢ gcd n n = n")
        print(f"Openrouter result: {result}")
    except Exception as e:
        print(f"Openrouter test failed: {e}")
    # Test fireworks
    try:
        fw_runner = UnifiedAPIRunner(
            provider="fireworks",
            model="accounts/fireworks/models/deepseek-v3p2",
            num_samples=3,
            reasoning_enabled=False
        )
        result = fw_runner.generate("n : ℕ\n⊢ gcd n n = n")
        print(f"Fireworks result: {result}")
    except Exception as e:
        print(f"Fireworks test failed: {e}")
