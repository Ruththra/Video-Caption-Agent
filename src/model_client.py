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
import time
from typing import Any, List, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger("model_client")

MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", "120"))
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
# Default is a Fireworks-hosted, open-weight, vision-capable model. Confirm the
# exact allowed model slug for Track 2 in the participant guide / Discord
# (ALLOWED_MODELS, if the harness enforces one) before your scored run.
DEFAULT_MODEL_NAME = "accounts/fireworks/models/gemma3-27b-it"
DEBUG_MODEL_RAW = os.environ.get("DEBUG_MODEL_RAW", "0").lower() in {"1", "true", "yes", "on"}
MODEL_RAW_MAX_CHARS = int(os.environ.get("MODEL_RAW_MAX_CHARS", "4000"))
MODEL_RAW_LOG_DIR = os.environ.get("MODEL_RAW_LOG_DIR")


class ModelError(RuntimeError):
    pass


def _clip_text(text: str, limit: int = MODEL_RAW_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n... <truncated {len(text) - limit} chars> ...\n{text[-half:]}"


def _safe_json_preview(obj: Any, limit: int = MODEL_RAW_MAX_CHARS) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:  # noqa: BLE001
        text = str(obj)
    return _clip_text(text, limit)


def _maybe_write_raw_response(model_name: str, data: dict) -> None:
    """Optional debug preservation for raw Fireworks responses.

    Set MODEL_RAW_LOG_DIR=/tmp/model_raw to preserve full model responses as
    JSON files. This intentionally does not log request headers or API keys.
    """
    if not MODEL_RAW_LOG_DIR:
        return
    try:
        os.makedirs(MODEL_RAW_LOG_DIR, exist_ok=True)
        safe_model = model_name.replace("/", "_").replace(":", "_")
        path = os.path.join(MODEL_RAW_LOG_DIR, f"fireworks_{safe_model}_{time.time_ns()}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Preserved raw model response at %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not preserve raw model response: %s", exc)


def _content_part_to_text(part: Any) -> str:
    """Normalize OpenAI/Fireworks-style message content into plain text.

    Kimi/Fireworks may return message.content as a string, a list of structured
    parts, a dict, or occasionally an empty content field plus auxiliary fields.
    Downstream code should always receive a string.
    """
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, list):
        chunks = [_content_part_to_text(p).strip() for p in part]
        return "\n".join(c for c in chunks if c)
    if isinstance(part, dict):
        # Common structured text forms.
        for key in ("text", "content", "output_text", "value"):
            value = part.get(key)
            if isinstance(value, (str, list, dict)):
                text = _content_part_to_text(value).strip()
                if text:
                    return text
        # Skip image echo parts, but keep unknown non-empty structures so the
        # caller can still inspect/debug them.
        if part.get("type") in {"image_url", "input_image"}:
            return ""
        return json.dumps(part, ensure_ascii=False, default=str)
    return str(part)


class ModelClient:
    """Base interface. `caption(images_b64, prompt)` returns raw text."""

    def caption(self, images_b64: List[str], prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class FireworksClient(ModelClient):
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
        self.last_raw_response: Optional[dict] = None
        self.last_message_text: str = ""
        if not self.api_key:
            raise ModelError(
                "FIREWORKS_API_KEY is not set. Use MODEL_PROVIDER=mock for local "
                "testing without credentials."
            )

    @retry(
        reraise=True,
        retry=retry_if_exception_type((requests.RequestException, ModelError)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    def _post(self, payload: dict) -> dict:
        try:
            resp = requests.post(
                f"{FIREWORKS_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=MODEL_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise ModelError(f"Fireworks request failure: {exc}") from exc

        if resp.status_code >= 400:
            raise ModelError(f"Fireworks API error {resp.status_code}: {resp.text[:1000]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ModelError(f"Fireworks returned non-JSON HTTP response: {resp.text[:1000]}") from exc

    def _message_to_text(self, choice: dict) -> str:
        message = choice.get("message") or {}
        text = _content_part_to_text(message.get("content")).strip()
        if text:
            return text

        # Some providers put generated text in auxiliary fields when content is
        # empty/null, especially on newer multimodal/chat backends.
        for key in ("reasoning_content", "tool_calls", "function_call"):
            text = _content_part_to_text(message.get(key)).strip()
            if text:
                return text
        for key in ("text", "content", "output_text"):
            text = _content_part_to_text(choice.get(key)).strip()
            if text:
                return text
        return ""

    def caption(self, images_b64: List[str], prompt: str, json_mode: bool = False) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 900,
            "temperature": 0.25,
        }

        # Kimi image calls may reject response_format=json_object. Keep strict
        # JSON mode for text-only style-rewrite calls, but avoid it when images
        # are present. The captioning parser now handles plain text robustly.
        if json_mode and not images_b64:
            payload["response_format"] = {"type": "json_object"}

        try:
            data = self._post(payload)
        except ModelError as exc:
            # Some models reject response_format even for text-only calls. Fall
            # back once without it while preserving the real error if that also
            # fails.
            if json_mode and "response_format" in str(exc).lower():
                logger.warning("Fireworks rejected response_format; retrying once without JSON mode: %s", exc)
                payload.pop("response_format", None)
                data = self._post(payload)
            else:
                raise
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Fireworks request failure: {exc}") from exc

        self.last_raw_response = data
        _maybe_write_raw_response(self.model_name, data)
        if DEBUG_MODEL_RAW:
            logger.warning("Raw Fireworks response preview:\n%s", _safe_json_preview(data))

        try:
            choices = data.get("choices") or []
            if not choices:
                raise ModelError(f"Fireworks response had no choices: {_safe_json_preview(data, 1000)}")
            text = self._message_to_text(choices[0]).strip()
            self.last_message_text = text
            if text:
                if DEBUG_MODEL_RAW:
                    logger.warning("Extracted model text preview:\n%s", _clip_text(text))
                return text

            # Empty content is actionable debug info. Return a compact response
            # preview rather than silently producing an empty string.
            preview = _safe_json_preview(choices[0], 1500)
            logger.warning("Fireworks response had empty message.content; choice preview:\n%s", preview)
            return preview
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"Fireworks response parse failure: {exc}; raw={_safe_json_preview(data, 1500)}") from exc


class MockClient(ModelClient):
    """Deterministic offline stand-in so the pipeline is fully testable
    without any API key. Returns a plausible-shaped JSON string so the
    downstream parser exercises the exact same code path as a real call."""

    def caption(self, images_b64: List[str], prompt: str, json_mode: bool = False) -> str:
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

    def caption(self, images_b64: List[str], prompt: str, json_mode: bool = False) -> str:
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
            "setting": "scene derived from local BLIP frame captions",
            "subjects": [],
            "actions": [],
            "objects": [],
            "mood": "neutral",
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
