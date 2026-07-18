"""Qwen3 OpenAI-compatible API chat wrapper."""

from __future__ import annotations

import importlib.util
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Qwen3ModelError(Exception):
    """Raised when Qwen3 API initialization or generation fails."""


class Qwen3Model:
    """Text-only Qwen3 client for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        model_name: str = "Qwen3-30B-A3B-Instruct-2507",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        strip_thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.strip_thinking = strip_thinking
        self.kwargs = kwargs

        if importlib.util.find_spec("openai") is None:
            raise Qwen3ModelError(
                "openai is required for Qwen3 API inference. Install it with `pip install openai`."
            )
        if not self.base_url:
            raise Qwen3ModelError("base_url is required for Qwen3 API inference, for example https://xxx/v1.")
        if not self.api_key:
            raise Qwen3ModelError("api_key is required for Qwen3 API inference.")

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("Initialized Qwen3 API client for %s at %s", self.model_name, self.base_url)

    @staticmethod
    def _messages(prompt: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if not prompt:
            raise Qwen3ModelError("messages must not be empty")
        for idx, message in enumerate(prompt):
            if "role" not in message or "content" not in message:
                raise Qwen3ModelError(f"message at index {idx} must contain 'role' and 'content'")
        return prompt

    def _clean(self, text: str) -> str:
        text = text.strip()
        if self.strip_thinking:
            text = re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", text).strip()
        return text

    def generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        messages = self._messages(prompt)
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model_name", kwargs.get("model", self.model_name)),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
        except Exception as exc:
            raise Qwen3ModelError(f"Qwen3 API generation failed: {exc}") from exc

        if not response.choices:
            raise Qwen3ModelError("Qwen3 API returned no choices")
        content = response.choices[0].message.content
        if content is None:
            raise Qwen3ModelError("Qwen3 API returned an empty message content")
        return self._clean(content)

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, str]]]], **kwargs: Any) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in batch_prompts]
