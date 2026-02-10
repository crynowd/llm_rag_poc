import requests
from typing import List, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def embed_one(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        r = requests.post(url, json={"model": model, "prompt": text}, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data["embedding"]

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Нестрриминговый вызов /api/generate.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system:
            payload["system"] = system

        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

