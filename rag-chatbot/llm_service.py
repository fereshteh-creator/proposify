# llm_service.py

import os
import time
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError

# Make sure .env is loaded as soon as this module is imported
load_dotenv()


class LLMService:
    def __init__(self) -> None:
        """Initialize the LLM service with API credentials from environment."""
        api_key = os.getenv("BFH_LLM_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Make sure .env exists and contains BFH_LLM_API_KEY"
            )

        self.client = OpenAI(
            base_url="https://inference.mlmp.ti.bfh.ch/api",
            api_key=api_key,
        )

        print("LLM Service initialized successfully.")

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate a chat-style completion from the BFH LLM.

        Returns a dict with keys: text, model, usage.
        """
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="ollama/gpt-oss:120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    timeout=180,
                )
                return {
                    "text": response.choices[0].message.content.strip(),
                    "model": response.model,
                    "usage": response.usage,
                }
            except (APITimeoutError, APIConnectionError) as err:
                last_err = err
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "The BFH LLM could not be reached (timeout/connection). "
                    "Please retry with a smaller file or try again later."
                ) from err
        raise last_err or RuntimeError("Unknown LLM error")


llm_service = LLMService()

