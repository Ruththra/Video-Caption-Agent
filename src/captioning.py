"""
Two-stage captioning:
  1. Grounded factual scene summary from chronological frames.
  2. Style-conditioned captions derived ONLY from that summary (not the raw
     frames again) -- this keeps every style anchored to the same observed
     facts instead of letting the model re-imagine details per style.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

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

Return ONLY a JSON object with exactly these keys:
{{
  "setting": "short description of location/environment",
  "subjects": ["list of visible people/animals/main subjects"],
  "actions": ["list of visible actions/events, in order if sequential"],
  "objects": ["list of notable visible objects"],
  "mood": "one or two words for the overall visual mood/tone of the scene",
  "notable_details": "any other grounded detail worth keeping, or empty string"
}}
No prose outside the JSON."""

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
e.g. {{"formal": "...", "sarcastic": "..."}}. No prose outside the JSON."""


def _extract_json_object(text: str) -> dict:
    """Best-effort JSON extraction: try direct parse, then find the first
    balanced {...} block, then give up."""
    text = text.strip()
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError("unbalanced JSON braces in model output")


def get_scene_summary(client: ModelClient, images_b64: List[str]) -> dict:
    prompt = SUMMARY_PROMPT.format(n=len(images_b64))
    raw = client.caption(images_b64, prompt)
    try:
        return _extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scene summary JSON parse failed (%s); retrying once", exc)
        raw_retry = client.caption(
            images_b64, prompt + "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Return ONLY the JSON object, nothing else."
        )
        try:
            return _extract_json_object(raw_retry)
        except Exception:
            # Last-resort minimal summary so the pipeline can still produce
            # template-based fallback captions rather than failing the task.
            return {
                "setting": "unspecified",
                "subjects": [],
                "actions": [],
                "objects": [],
                "mood": "neutral",
                "notable_details": raw[:200] if isinstance(raw, str) else "",
            }


def get_style_captions(client: ModelClient, summary: dict, styles: List[str]) -> Dict[str, str]:
    prompt = STYLE_PROMPT_TEMPLATE.format(
        styles=", ".join(styles), summary_json=json.dumps(summary, ensure_ascii=False)
    )
    try:
        raw = client.caption([], prompt)  # text-only call, no images needed for this stage
    except ModelError as exc:
        logger.warning("style caption model call failed (%s); using template fallback", exc)
        return {}

    try:
        parsed = _extract_json_object(raw)
        return {k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("style caption JSON parse failed (%s); retrying once", exc)
        try:
            raw_retry = client.caption(
                [], prompt + "\n\nIMPORTANT: your previous response was not valid JSON. "
                "Return ONLY the JSON object, nothing else."
            )
            parsed = _extract_json_object(raw_retry)
            return {k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()}
        except Exception:
            return {}


def template_fallback_caption(summary: dict, style: str) -> str:
    """Deterministic, non-hallucinating fallback used when the model
    couldn't produce a valid caption for a requested style."""
    setting = summary.get("setting") or "an unspecified setting"
    subjects = summary.get("subjects") or []
    actions = summary.get("actions") or []
    subj_txt = ", ".join(subjects) if subjects else "the visible subject(s)"
    act_txt = ", ".join(actions) if actions else "an ongoing scene"

    base = f"{subj_txt} in {setting}, {act_txt}."
    templates = {
        "formal": f"The clip shows {base}",
        "sarcastic": f"Well, would you look at that: {base} Riveting.",
        "humorous_tech": f"System log: detected {base} No errors thrown, humor.exe not found.",
        "humorous_non_tech": f"Just your everyday moment: {base} Nothing to see here, folks.",
    }
    return templates.get(style, f"A short clip showing {base}")


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
