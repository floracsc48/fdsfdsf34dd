"""Voice assistant utilities for interacting with the DeepSeek API."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import logging

import requests


logger = logging.getLogger(__name__)


class SpeechListener:
    """Handle user speech, keep conversation history and talk to DeepSeek."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        reset_commands: Optional[Iterable[str]] = None,
        timeout: Optional[float] = 30.0,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

        self._base_history: List[Dict[str, str]] = []
        if system_prompt:
            self._base_history.append({"role": "system", "content": system_prompt})

        # Ensure chat history is preserved between multiple calls/sessions.
        self.chat_history: List[Dict[str, str]] = list(self._base_history)

        self.reset_commands = set(command.lower() for command in (reset_commands or []))
        if not self.reset_commands:
            self.reset_commands = {"reset", "очистить историю", "очистить контекст"}

    def clear_history(self) -> None:
        """Reset the conversation while keeping the base system prompt."""

        logger.debug("Clearing chat history")
        self.chat_history = list(self._base_history)

    def _should_reset(self, user_text: str) -> bool:
        return user_text.strip().lower() in self.reset_commands

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def send_to_deepseek(self, user_text: str) -> str:
        """Send a message to DeepSeek and update the running chat history."""

        messages = list(self.chat_history)
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model,
            "messages": messages,
        }

        logger.debug("Sending payload to DeepSeek: %s", payload)
        response = self.session.post(
            self.api_url,
            headers=self._build_headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        try:
            assistant_text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as error:
            logger.error("Unexpected response structure from DeepSeek: %s", data)
            raise ValueError("Unexpected response structure from DeepSeek") from error

        # Persist both the user and assistant messages for subsequent requests.
        self.chat_history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )

        logger.debug("Updated chat history: %s", self.chat_history)
        return assistant_text

    def process(self, user_text: str) -> str:
        """Process new user input, optionally clearing history before sending it."""

        normalized = user_text.strip()
        if not normalized:
            return ""

        if self._should_reset(normalized):
            self.clear_history()
            return "История диалога очищена."

        return self.send_to_deepseek(normalized)

