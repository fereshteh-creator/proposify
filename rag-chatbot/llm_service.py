# llm_service.py

import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from together import Together
from together.error import APIConnectionError, Timeout, APIError

# Make sure .env is loaded as soon as this module is imported
load_dotenv()


class LLMService:
    """
    Thin wrapper around the Together chat API.

    The effective model is chosen at call time based on the current
    Streamlit session state (if available). When no session is active,
    we fall back to GPT‑OSS 120B.
    """

    def __init__(self) -> None:
        api_key = os.getenv("TOGETHER_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Make sure .env exists and contains TOGETHER_API_KEY"
            )

        self.client = Together(api_key=api_key)

    @staticmethod
    def _current_model() -> str:
        """
        Decide which Together model to call.

        Preference order:
        - Streamlit session state's `llm_model` (if set)
        - Environment variable DEFAULT_LLM_MODEL
        - GPT‑OSS 120B as a safe default
        """
        # 1) Try to read from Streamlit session (app context)
        try:
            import streamlit as st  # type: ignore

            choice = getattr(st.session_state, "llm_model", "") or ""
        except Exception:
            choice = ""

        choice = str(choice).strip()

        if choice == "Qwen3 80B (Together)":
            return "Qwen/Qwen3-Next-80B-A3B-Instruct"
        if choice == "GPT-OSS 120B (Together)":
            return "openai/gpt-oss-120b"

        # 2) Fallback: environment variable
        env_choice = os.getenv("DEFAULT_LLM_MODEL", "").strip()
        if env_choice:
            return env_choice

        # 3) Hard default
        return "openai/gpt-oss-120b"

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        model: str | None = None,
    ) -> Dict[str, Any]:
        """
        Generate a chat-style completion via Together.

        Returns a dict with keys: text, model, usage.
        """
        last_err: Exception | None = None
        model_name = model or self._current_model()

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
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
                    "usage": getattr(response, "usage", {}),
                }
            except (Timeout, APIConnectionError, APIError) as err:
                last_err = err
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "The LLM backend could not be reached (timeout/connection). "
                    "Please retry with a smaller file or try again later."
                ) from err

        raise last_err or RuntimeError("Unknown LLM error")


llm_service = LLMService()
