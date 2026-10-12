"""Ollama HTTP API istemcisi.

Sunucu adresi config.json'dan gelir; OLLAMA_HOST ortam değişkeni onu ezer.
"""
import base64
import json

import requests

from .config import CFG

DEFAULT_HOST = CFG["ollama_host"]


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: int = 600):
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        format_schema: dict | str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ) -> str:
        """Tek mesajlık chat çağrısı. images: dosya yolları (base64'e çevrilir)."""
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [_b64(p) for p in images]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(message)

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        if format_schema is not None:
            payload["format"] = format_schema  # dict => JSON schema, "json" => serbest JSON

        resp = requests.post(
            f"{self.host}/api/chat", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.host}/api/tags", timeout=30)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def parse_json_loose(text: str) -> dict:
    """Model çıktısındaki JSON'u toleranslı biçimde ayıkla."""
    text = text.strip()
    if text.startswith("```"):
        # ```json ... ``` bloğunu soy
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0] if "```" in text else text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Çıktıda JSON bulunamadı: {text[:200]}")
    return json.loads(text[start : end + 1])
