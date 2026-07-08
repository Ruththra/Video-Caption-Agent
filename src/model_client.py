"""
Pluggable vision-model client.

MODEL_PROVIDER=fireworks  (default, required for actual Track 2 scoring --
                            the hackathon brief states Track 2 compute is
                            "Fireworks AI API")
MODEL_PROVIDER=mock       (no network/model calls at all -- for CI / schema
                            testing without any API key)
MODEL_PROVIDER=local      (optional open-weight fallback for offline dev on
                            a machine with no Fireworks access; NOT what you
                            submit for the leaderboard run, see README)
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("model_client")

MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", "120"))
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
# Default is a Fireworks-hosted, open-weight, vision-capable model. Confirm the
# exact allowed model slug for Track 2 in the participant guide / Discord
# (ALLOWED_MODELS, if the harness enforces one) before your scored run --
# Gemma 3 27B is a good default because it is also eligible for the
# separate "Best Use of Gemma" Track 2 prize.
DEFAULT_MODEL_NAME = "accounts/fireworks/models/gemma3-27b-it"


class ModelError(RuntimeError):
    pass


class ModelClient:
    """Base interface. `caption(images_b64, prompt)` returns raw text."""

    def caption(self, images_b64: List[str], prompt: str) -> str:
        raise NotImplementedError


class FireworksClient(ModelClient):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        if not self.api_key:
            raise ModelError(
                "FIREWORKS_API_KEY is not set. Use MODEL_PROVIDER=mock for local "
                "testing without credentials."
            )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _post(self, payload: dict) -> dict:
        resp = requests.post(
            f"{FIREWORKS_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 400:
            raise ModelError(f"Fireworks API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def caption(self, images_b64: List[str], prompt: str, json_mode: bool = True) -> str:
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 900,
            "temperature": 0.4,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            data = self._post(payload)
            return data["choices"][0]["message"]["content"]
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Fireworks request/parse failure: {exc}") from exc


class MockClient(ModelClient):
    """Deterministic offline stand-in so the pipeline is fully testable
    without any API key. Returns a plausible-shaped JSON string so the
    downstream parser exercises the exact same code path as a real call."""

    def caption(self, images_b64: List[str], prompt: str) -> str:
        if "scene summary" in prompt.lower() or "factual" in prompt.lower():
            return json.dumps({
                "setting": "an outdoor urban scene",
                "subjects": ["a person"],
                "actions": ["walking"],
                "objects": ["a street", "buildings"],
                "mood": "neutral",
                "notable_details": f"{len(images_b64)} sampled frames analyzed in mock mode",
            })
        return json.dumps({
            "formal": "A person is seen walking through an urban street setting.",
            "sarcastic": "Riveting stuff: someone walked. On a street. Groundbreaking.",
            "humorous_tech": "Detected: one (1) biped executing a WALK() loop, no exceptions thrown.",
            "humorous_non_tech": "Just a person out for a stroll, living their best pedestrian life.",
        })


class LocalClient(ModelClient):
    """Optional open-weight local fallback (BLIP image captioning).

    This is intentionally simple and single-frame-oriented: BLIP is an
    image captioner, not a video model, so we caption a small subset of
    frames and merge them into a pseudo scene-summary. It exists as a
    genuine no-API-key fallback path, not as the primary submission
    strategy for Track 2 (see README: Track 2 compute is Fireworks AI API).
    Requires requirements-local.txt to be installed; not part of the
    default Docker image.
    """

    def __init__(self):
        try:
            import torch  # noqa: F401
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise ModelError(
                "MODEL_PROVIDER=local requires torch+transformers "
                "(pip install -r requirements-local.txt)"
            ) from exc
        self._torch = __import__("torch")
        self._BlipProcessor = BlipProcessor
        self._BlipForConditionalGeneration = BlipForConditionalGeneration
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def caption(self, images_b64: List[str], prompt: str) -> str:
        from io import BytesIO
        from PIL import Image

        is_summary_call = "scene summary" in prompt.lower() or "factual" in prompt.lower()
        if not is_summary_call:
            # Style rewriting has no local vision component -- signal the
            # caller to use the template-based fallback in captioning.py.
            raise ModelError("LocalClient only supports the scene-summary stage")

        # Sample at most 4 frames to keep CPU inference time bounded.
        subset = images_b64[:: max(1, len(images_b64) // 4)][:4] or images_b64[:1]
        captions = []
        for b64 in subset:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            inputs = self.processor(img, return_tensors="pt")
            out = self.model.generate(**inputs, max_new_tokens=40)
            captions.append(self.processor.decode(out[0], skip_special_tokens=True))

        return json.dumps({
            "setting": "unspecified (derived from local BLIP frame captions)",
            "subjects": [],
            "actions": [],
            "objects": [],
            "mood": "unspecified",
            "notable_details": " | ".join(captions),
        })


def build_client() -> ModelClient:
    provider = os.environ.get("MODEL_PROVIDER", "fireworks").lower().strip()
    if provider == "fireworks":
        return FireworksClient()
    if provider == "mock":
        return MockClient()
    if provider == "local":
        return LocalClient()
    raise ModelError(f"Unknown MODEL_PROVIDER: {provider!r} (expected fireworks|mock|local)")


def encode_jpeg_files(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(base64.b64encode(f.read()).decode("ascii"))
    return out
