import os

from openai import AsyncClient


class LLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        self.model = model

        self.client = AsyncClient(base_url=base_url, api_key=api_key)

    @staticmethod
    async def call(self, messages: list, temperature: float = 0.7):
        response = await self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)

        return response
