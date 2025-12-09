from abc import ABC, abstractmethod
from typing import List
import re
import requests
from openai import OpenAI


class APIClient(ABC):
    """Base class for LLM provider clients"""

    @abstractmethod
    def generate_tactics(self, state: str) -> List[str]:
        """Generate tactics from proof state"""
        pass

    def create_prompt(self, state: str) -> str:
        """Tactic suggestion prompt template"""
        return (
            f"Help me prove a theorem in Lean 4. Here is my current proof state:\n{state}\n"
            "Based on this state, suggest the next tactic I should use in Lean 4 code. Only output one tactic step in lean code and nothing else."
        )

    def extract_tactic(self, response: str) -> str:
        """Extract tactic from response, handles both fence and no fence cases"""
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

    def deduplicate(self, suggestions: List[str]) -> List[str]:
        """Keep list of only unique tactics"""
        seen = set()
        unique = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique.append(suggestion)
        return unique


class OpenRouterClient(APIClient):
    """OpenRouter client (for benchmarking only)"""

    def __init__(self, model: str, api_key: str, num_samples: int = 10):
        self.model = model
        self.api_key = api_key
        self.temperature = 1.0
        self.max_tokens = 1024
        self.num_samples = num_samples # choices per request

    def generate_tactics(self, state: str) -> List[str]:
        """Generate tactics based on current proof state"""
        prompt = self.create_prompt(state)
        # print(prompt)
        messages = [
            # {"role": "system", "content": "Help the user with the next step of their proof in Lean 4. Never output anything other than Lean 4 code."},
            {"role": "user", "content": prompt}
        ]
        suggestions = []

        for sample in range(self.num_samples):
            try:
                url = "https://openrouter.ai/api/v1/chat/completions"
                payload = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "reasoning":{
                        "enabled": True
                    },
                    "messages": messages
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                raw = requests.post(url, json=payload, headers=headers)
                response = raw.json()

                # Debug
                if 'choices' not in response:
                    print(f"Unexpected API response: {response}")

                # print(response['choices'][0]['message']['content'])
                tactic = self.extract_tactic(response['choices'][0]['message']['content'])
                if tactic:
                    suggestions.append(tactic)
                    # print(tactic)

                # print(f"Input tokens: response['usage']['prompt_tokens']")
                # print(f"Output tokens: response['usage']['completion_tokens']")

            except Exception as e:
                print(f"API call failed: {e}")

        to_return = self.deduplicate(suggestions)
        # print(f"Generated {len(to_return)}/{self.num_samples} unique tactic suggestions")
        return to_return


class FireworksClient(APIClient):
    """Fireworks client (benchmark custom or fine-tuned models)"""

    def __init__(self, model: str, api_key: str, num_samples: int = 10):
        self.model = model
        self.api_key = api_key
        self.temperature = 1.0
        self.max_tokens = 1024
        self.num_samples = num_samples # choices per request

    def generate_tactics(self, state: str) -> List[str]:
        """Generate tactics based on current proof state"""
        prompt = self.create_prompt(state)
        # print(prompt)
        messages = [
            # {"role": "system", "content": "Help the user with the next step of their proof in Lean 4. Never output anything other than Lean 4 code."},
            {"role": "user", "content": prompt}
        ]
        suggestions = []

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )

        for sample in range(self.num_samples):
            try:
                response = client.chat.completions.create(
                    model=self.model, 
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    reasoning_effort=True, # turn reasoning on
                    messages=messages
                )

                response = response.choices[0].message.content
                # print(response)

                # print(response['choices'][0]['message']['content'])
                tactic = self.extract_tactic(response)
                if tactic:
                    suggestions.append(tactic)
                    # print(tactic)

                # print(f"Input tokens: response['usage']['prompt_tokens']")
                # print(f"Output tokens: response['usage']['completion_tokens']")

            except Exception as e:
                print(f"API call failed: {e}")

        to_return = self.deduplicate(suggestions)
        # print(f"Generated {len(to_return)}/{self.num_samples} unique tactic suggestions")
        return to_return