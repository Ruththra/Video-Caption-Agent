"""
Two-stage captioning:
  1. Grounded factual scene summary from chronological frames.
  2. Style-conditioned captions derived ONLY from that summary (not the raw
     frames again) -- this keeps every style anchored to the same observed
     facts instead of letting the model re-imagine details per style.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List

from model_client import ModelClient, ModelError

logger = logging.getLogger("captioning")

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

SUMMARY_PROMPT = """You are a meticulous video analyst. You are shown {n} frames sampled \
in chronological order across the FULL duration of a short video clip (not just the \
start). Look at all frames together as one sequence and produce a grounded, factual \
scene summary.

Rules:
- Only describe what is clearly visible. Never guess names, identities, locations, \
brands, or causes that are not visually evident.
- If something is ambiguous, describe it generically (e.g. "a person" not "a man named X").
- Note any change or motion across the frame sequence if visible (e.g. camera pans, \
subject moves, action progresses).
- Keep it factual, not stylistic.

Preferred output: return ONLY a JSON object with exactly these keys:
{{
  "setting": "short description of location/environment",
  "subjects": ["list of visible people/animals/main subjects"],
  "actions": ["list of visible actions/events, in order if sequential"],
  "objects": ["list of notable visible objects"],
  "mood": "one or two words for the overall visual mood/tone of the scene",
  "notable_details": "one factual paragraph with concrete visual details"
}}

If you cannot return valid JSON, return ONLY one plain factual paragraph beginning
VISUAL_DESCRIPTION: followed by the visible scene details. No markdown, no bullets."""

STYLE_PROMPT_TEMPLATE = """You are a caption writer. Using ONLY the grounded scene \
summary below (do not invent new facts, objects, people, or actions that are not in \
it), write a one-sentence caption for each of the following styles: {styles}.

Style definitions:
- formal: professional, objective, factual tone.
- sarcastic: dry, ironic, lightly mocking, but still grounded and not mean-spirited.
- humorous_tech: funny, references programming/software/AI/debugging/hardware, but \
still understandable to a non-engineer.
- humorous_non_tech: funny using everyday language, no technical jargon at all.

Rules:
- Each caption should be ONE sentence. Use a second short sentence only if the scene \
genuinely needs it to stay accurate.
- Never contradict the scene summary or add unseen details.
- Only write captions for the styles requested: {styles}.

Scene summary (JSON):
{summary_json}

Return ONLY a JSON object mapping each requested style name to its caption string, \
e.g. {{"formal": "...", "sarcastic": "..."}}. No prose outside the JSON.
If JSON is impossible, use one line per style in the form `style: caption`."""


def _clean_model_text(text: Any, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            text = str(text)
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    text = re.sub(r"^VISUAL_DESCRIPTION\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _load_json_lenient(candidate: str) -> dict:
    candidate = candidate.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = candidate
    # Common LLM slips: smart quotes and trailing commas.
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Last deterministic repair: parse Python-ish dicts with single quotes.
    try:
        parsed = ast.literal_eval(repaired)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    raise ValueError("candidate is not a JSON object")


def _extract_json_object(text: str) -> dict:
    """Best-effort JSON extraction: direct parse, repaired parse, then the
    first balanced {...} block."""
    text = text.strip()
    try:
        return _load_json_lenient(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return _load_json_lenient(candidate)
    raise ValueError("unbalanced JSON braces in model output")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return _clean_model_text(value, max_chars=500)
    if isinstance(value, list):
        return ", ".join(_clean_model_text(v, max_chars=160) for v in value if _clean_model_text(v, max_chars=160))
    if value is None:
        return ""
    return _clean_model_text(value, max_chars=500)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            txt = _clean_model_text(item, max_chars=160)
            if txt:
                out.append(txt)
        return out
    if isinstance(value, str):
        text = _clean_model_text(value, max_chars=400)
        if not text:
            return []
        # Split only simple delimiter lists; keep full sentences intact.
        if ";" in text or "," in text and len(text.split()) <= 12:
            return [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
        return [text]
    return []


def _plain_text_to_summary(raw: Any) -> dict:
    details = _clean_model_text(raw)
    if not details:
        details = "The sampled frames show a visible scene, but the model did not provide readable details."
    return {
        "setting": "the scene shown in the sampled frames",
        "subjects": ["visible subject(s)"],
        "actions": ["visible activity in the clip"],
        "objects": [],
        "mood": "neutral",
        "notable_details": details,
    }


def _coerce_scene_summary(parsed: dict, raw: Any = "") -> dict:
    """Normalize any dict-like model output into the required summary schema."""
    raw_details = _clean_model_text(raw)
    details_candidates = [
        parsed.get("notable_details"),
        parsed.get("description"),
        parsed.get("visual_description"),
        parsed.get("summary"),
        parsed.get("caption"),
    ]
    notable_details = next((_as_text(v) for v in details_candidates if _as_text(v)), "")
    if not notable_details and raw_details and len(raw_details) < 1000:
        notable_details = raw_details

    summary = {
        "setting": _as_text(parsed.get("setting") or parsed.get("location") or parsed.get("environment"))
        or "the scene shown in the sampled frames",
        "subjects": _as_list(parsed.get("subjects") or parsed.get("people") or parsed.get("main_subjects"))
        or ["visible subject(s)"],
        "actions": _as_list(parsed.get("actions") or parsed.get("events") or parsed.get("motion"))
        or ["visible activity in the clip"],
        "objects": _as_list(parsed.get("objects") or parsed.get("items")),
        "mood": _as_text(parsed.get("mood") or parsed.get("tone")) or "neutral",
        "notable_details": notable_details,
    }
    return summary


def _call_caption(client: ModelClient, images_b64: List[str], prompt: str, json_mode: bool = False) -> str:
    try:
        return client.caption(images_b64, prompt, json_mode=json_mode)
    except TypeError:
        # Backward compatibility for a custom client that still implements the
        # original two-argument interface.
        return client.caption(images_b64, prompt)  # type: ignore[misc]


def get_scene_summary(client: ModelClient, images_b64: List[str]) -> dict:
    prompt = SUMMARY_PROMPT.format(n=len(images_b64))
    raw = _call_caption(client, images_b64, prompt, json_mode=False)
    try:
        return _coerce_scene_summary(_extract_json_object(raw), raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scene summary JSON parse failed (%s); retrying once", exc)
        logger.info("scene summary raw fallback preview: %s", _clean_model_text(raw, max_chars=700))
        raw_retry = _call_caption(
            client,
            images_b64,
            prompt + "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Return ONLY the JSON object. If you cannot, return VISUAL_DESCRIPTION: plus one factual paragraph.",
            json_mode=False,
        )
        try:
            return _coerce_scene_summary(_extract_json_object(raw_retry), raw_retry)
        except Exception as retry_exc:  # noqa: BLE001
            logger.warning("scene summary retry was still not JSON (%s); using raw visual text", retry_exc)
            # Crucial fix: use model free-form visual description as the
            # grounding detail instead of dropping to "unspecified".
            return _plain_text_to_summary(raw_retry or raw)


def _parse_style_lines(raw: str, styles: List[str]) -> Dict[str, str]:
    text = _clean_model_text(raw, max_chars=2500)
    if not text:
        return {}

    found: Dict[str, str] = {}
    style_alt = "|".join(re.escape(s) for s in styles)
    pattern = re.compile(
        rf"(?P<style>{style_alt})\s*[:\-–]\s*(?P<caption>.*?)(?=(?:\b(?:{style_alt})\b\s*[:\-–])|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        style = next((s for s in styles if s.lower() == match.group("style").lower()), None)
        caption = _clean_model_text(match.group("caption"), max_chars=350)
        if style and caption:
            found[style] = caption
    return found


def get_style_captions(client: ModelClient, summary: dict, styles: List[str]) -> Dict[str, str]:
    prompt = STYLE_PROMPT_TEMPLATE.format(
        styles=", ".join(styles), summary_json=json.dumps(summary, ensure_ascii=False)
    )
    try:
        raw = _call_caption(client, [], prompt, json_mode=True)  # text-only; JSON mode is safer here
    except ModelError as exc:
        logger.warning("style caption model call failed (%s); using template fallback", exc)
        return {}

    try:
        parsed = _extract_json_object(raw)
        return {k: _clean_model_text(v, max_chars=350) for k, v in parsed.items() if isinstance(v, str) and v.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("style caption JSON parse failed (%s); retrying once", exc)
        line_parsed = _parse_style_lines(raw, styles)
        if line_parsed:
            return line_parsed
        try:
            raw_retry = _call_caption(
                client,
                [],
                prompt + "\n\nIMPORTANT: your previous response was not valid JSON. "
                "Return ONLY the JSON object. If JSON is impossible, use `style: caption` lines.",
                json_mode=True,
            )
            try:
                parsed = _extract_json_object(raw_retry)
                return {k: _clean_model_text(v, max_chars=350) for k, v in parsed.items() if isinstance(v, str) and v.strip()}
            except Exception:
                return _parse_style_lines(raw_retry, styles)
        except Exception:
            return {}


def _summary_grounding_text(summary: dict) -> str:
    details = _clean_model_text(summary.get("notable_details"), max_chars=450)
    # Prefer real visual details whenever available. This is the critical path
    # that prevents "unspecified" fallback captions.
    if details and "message.content" not in details.lower():
        return details.rstrip(" .")

    setting = _as_text(summary.get("setting")) or "the scene shown in the sampled frames"
    subjects = summary.get("subjects") or []
    actions = summary.get("actions") or []
    subj_txt = ", ".join(subjects) if subjects else "visible subject(s)"
    act_txt = ", ".join(actions) if actions else "visible activity in the clip"
    return f"{subj_txt} in {setting}, with {act_txt}".rstrip(" .")


def _sentence(text: str) -> str:
    text = _clean_model_text(text, max_chars=500).strip()
    if not text:
        text = "The sampled frames show a visible scene"
    text = text[0].upper() + text[1:] if text else text
    if text[-1] not in ".!?":
        text += "."
    return text


def template_fallback_caption(summary: dict, style: str) -> str:
    """Deterministic, non-hallucinating fallback used when the model
    couldn't produce a valid caption for a requested style.

    The fallback now prioritizes `notable_details`, which may contain Kimi's
    free-form visual description when JSON parsing fails.
    """
    scene = _summary_grounding_text(summary)
    plain_scene = scene.rstrip(" .")

    templates = {
        "formal": _sentence(plain_scene),
        "sarcastic": _sentence(f"{plain_scene}. Truly, cinema has never looked so calmly observational"),
        "humorous_tech": _sentence(f"System log: visual input parsed as {plain_scene}; no hallucination module required"),
        "humorous_non_tech": _sentence(f"{plain_scene}, which is basically the scene saying, 'yep, this is happening'"),
    }
    return templates.get(style, _sentence(plain_scene))


def generate_captions_for_task(
    client: ModelClient, images_b64: List[str], styles: List[str]
) -> Dict[str, str]:
    summary = get_scene_summary(client, images_b64)
    model_captions = get_style_captions(client, summary, styles)

    result: Dict[str, str] = {}
    for style in styles:
        caption = model_captions.get(style)
        if not caption:
            caption = template_fallback_caption(summary, style)
        result[style] = caption.strip()

    # Guarantee the four required styles are always present, even if the
    # caller asked for a different/extra set of styles.
    for style in REQUIRED_STYLES:
        if style not in result:
            result[style] = template_fallback_caption(summary, style)

    return result
