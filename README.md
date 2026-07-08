# Video Captioning Agent — AMD Developer Hackathon: ACT II, Track 2

A Dockerized agent that reads `/input/tasks.json`, downloads each video, samples
frames across the **full** clip, generates grounded scene-accurate captions in
four styles (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`), and
writes `/output/results.json`.

## ⚠️ Important correction vs. an "open-source-first / run-it-locally" plan

I browsed the official hackathon page before building this
(https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii). For **Track 2**
it states explicitly:

> "Models are accessed via Fireworks AI API credits. Fine-tuning is explicitly
> permitted — you may also train your own captioner and use it alongside or
> instead of prompting... **Compute: Fireworks AI API**"

That means the track is scored around Fireworks-hosted inference, not a
locally-run open-weight model baked into the container. So this build uses
**Fireworks AI as the primary/scored path**, with open-weight *models hosted
on Fireworks* (Gemma 3 / Qwen-VL / Llama-Vision class), which still satisfies
"open-source/open-weight wherever possible" without fighting the rules. A
genuinely local fallback path is included for offline dev only — see
Strategy B below.

I could not access the full gated Participant Guide (registration/Discord-only,
behind the "Enroll" flow) or an exact `ALLOWED_MODELS` list for Track 2, so
**before your scored run, double check the Discord/participant guide for**:
- the exact list of allowed Fireworks model slugs for Track 2 (if enforced)
- exact harness env var names, if the judge harness sets its own
- exact submission deadline / leaderboard rerun schedule

Everything else below (input/output JSON shape, `/input` → `/output` contract,
Dockerized submission, public GitHub repo requirement, 30s–2min clips) is
taken directly from the public page content and this brief.

## 1. Architecture recommendation

**Two-stage, structured-output pipeline:**

1. **Frame sampling** — `ffprobe` gets duration, `ffmpeg` extracts ~8–16
   frames evenly spaced across the *whole* clip (not just frame 1).
2. **Stage A — grounded scene summary.** All frames go to the vision model
   once, in chronological order, with a prompt that forces a factual,
   non-speculative JSON summary (setting/subjects/actions/objects/mood).
3. **Stage B — style rewrite.** A second (image-free, cheap, fast) call takes
   *only* that JSON summary and rewrites it into the four requested styles in
   one structured JSON response.

Why this wins over "one big prompt, four captions at once":
- **Accuracy first** (per the brief's own priority order) — every style is
  anchored to one shared, grounded summary, so styles can't drift into
  different, possibly hallucinated, details.
- **Style matching** improves because Stage B is a pure text-rewriting task,
  which is what LLMs are best and most consistent at.
- **Cheaper/faster** — Stage B needs no image tokens, so you can retry it
  without re-paying for vision tokens.
- **Robust to judge JSON-strictness** — using `response_format:
  {"type":"json_object"}` plus a manual balanced-brace extractor plus one
  retry plus a deterministic template fallback means a style is *never*
  missing from `results.json`, even if the model misbehaves.

## 2. Chosen model / provider and backup

| | Primary | Backup |
|---|---|---|
| Provider | **Fireworks AI API** (`MODEL_PROVIDER=fireworks`) | Same provider, swap `MODEL_NAME` |
| Model | `accounts/fireworks/models/gemma3-27b-it` (multimodal, open-weight, Apache 2.0, also eligible for the separate "Best Use of Gemma" Track 2 prize) | `accounts/fireworks/models/qwen2p5-vl-32b-instruct` or `accounts/fireworks/models/llama-v3p2-90b-vision-instruct` if Gemma's vision quality underperforms on your test clips, or if it's excluded by an `ALLOWED_MODELS` list you find in Discord |
| Offline dev fallback | `MODEL_PROVIDER=local` — BLIP (`Salesforce/blip-image-captioning-base`), CPU-only, ~1GB, genuinely open-weight, no API key | n/a |
| No-key testing | `MODEL_PROVIDER=mock` — deterministic fake responses, exercises the exact same code path | n/a |

**Strategy A (hosted Fireworks endpoint) is the one to submit** — it's what
Track 2 compute is scored on, keeps the image small (<200MB compressed) and
build/startup fast, and lets you swap models with one env var if the model
turns out to be excluded or underperforms. **Strategy B (bundle/download a
local open model)** is kept only as `MODEL_PROVIDER=local`, useful if you want
to prototype without burning Fireworks credits, but don't rely on it for your
scored submission — it wasn't designed to compete with a real VLM's caption
quality, and Track 2's own compute line is Fireworks, not local GPU.

## 3. Final folder structure

```
video-caption-agent/
  Dockerfile
  requirements.txt          # runtime deps (small: requests, tenacity)
  requirements-local.txt    # optional, only for MODEL_PROVIDER=local
  requirements-dev.txt      # pytest, for `tests/`
  .dockerignore
  README.md
  src/
    main.py                 # entrypoint: tasks.json -> results.json
    video_utils.py           # download, ffprobe, ffmpeg frame sampling
    model_client.py          # Fireworks / mock / local providers
    captioning.py            # prompts, JSON parsing/repair, fallbacks
    schema.py                 # input/output validation
  sample_input/
    tasks.json
  tests/
    test_schema.py
```

## 4. Exact commands to run locally

```bash
cd video-caption-agent

# --- no API key needed, validates the whole pipeline shape ---
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q

mkdir -p /tmp/input /tmp/output
cp sample_input/tasks.json /tmp/input/tasks.json
TASKS_INPUT_PATH=/tmp/input/tasks.json \
RESULTS_OUTPUT_PATH=/tmp/output/results.json \
MODEL_PROVIDER=mock \
python3 src/main.py
python3 -m json.tool /tmp/output/results.json

# --- with a real Fireworks key ---
export FIREWORKS_API_KEY=sk-...
export MODEL_PROVIDER=fireworks
export MODEL_NAME=accounts/fireworks/models/gemma3-27b-it
TASKS_INPUT_PATH=/tmp/input/tasks.json \
RESULTS_OUTPUT_PATH=/tmp/output/results.json \
python3 src/main.py

# --- Docker build (linux/amd64) ---
docker build --platform linux/amd64 -t video-caption-agent:latest .

# --- Docker run ---
docker run --rm \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e MODEL_PROVIDER=fireworks \
  -v /tmp/input:/input \
  -v /tmp/output:/output \
  video-caption-agent:latest

# --- Docker run, mock mode, zero secrets required ---
docker run --rm \
  -e MODEL_PROVIDER=mock \
  -v /tmp/input:/input \
  -v /tmp/output:/output \
  video-caption-agent:latest

python3 -m json.tool /tmp/output/results.json
```

Expected output shape (`/output/results.json`):
```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## 5. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `fireworks` | `fireworks` \| `mock` \| `local` |
| `MODEL_NAME` | `accounts/fireworks/models/gemma3-27b-it` | Fireworks model slug |
| `FIREWORKS_API_KEY` | — | required for `fireworks` provider |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | override if the harness gives you a different base URL |
| `MAX_FRAMES` | `16` | cap on sampled frames per video |
| `MIN_FRAMES` | `8` | floor on sampled frames per video |
| `MODEL_TIMEOUT_SECONDS` | `120` | per-request timeout |
| `DOWNLOAD_TIMEOUT_SECONDS` | `60` | per-attempt video download timeout |
| `DOWNLOAD_RETRIES` | `3` | download retry attempts |
| `TASKS_INPUT_PATH` | `/input/tasks.json` | override for local testing |
| `RESULTS_OUTPUT_PATH` | `/output/results.json` | override for local testing |

Additional environment variables:

| `RUN_TIMEOUT_SECONDS` | `600` | Optional global run timeout (seconds). If set, the agent will write partial results and exit non-zero when the timeout elapses (default 600s = 10 minutes, matching Track 2 limit). |
| `DOWNLOAD_TIMEOUT_SECONDS` | `120` | Increased default per-attempt download timeout to 120s to be more robust on slow hosts. |

Implementation notes:
- `FireworksClient` now defensively serializes structured `content` responses to JSON strings so `captioning.py` can reliably parse model outputs even if the provider returns an already-parsed object.

## Error handling coverage

`missing tasks.json`, `malformed JSON`, `missing task_id`, `missing video_url`,
`empty styles`, `video download failure`, `ffmpeg/ffprobe failure`, `model
timeout`, `invalid model JSON response` are all caught per-task; the batch
always finishes and `/output/results.json` is always written with valid JSON
for every task, using deterministic, non-hallucinating template captions as a
last resort. The process exits `0` whenever it managed to write output —
verified in testing (see below).

## Testing performed

- `pytest tests/` — 12/12 schema tests pass.
- Synthetic 45s test video generated with `ffmpeg testsrc`, run through
  `video_utils.extract_frames` → 8 evenly spaced frames extracted successfully.
- Full `src/main.py` run against a 3-task `tasks.json` covering: (1) a
  reachable-video happy path, (2) an unreachable URL, (3) a task missing
  `task_id` — all three produced valid fallback/real captions, `results.json`
  passed `python -m json.tool`, and the process exited `0`.

## Final submission checklist

- [ ] `docker build --platform linux/amd64 -t <dockerhub-user>/video-caption-agent:latest .`
- [ ] `docker push <dockerhub-user>/video-caption-agent:latest` (public repo/tag)
- [ ] Public pull command in your README: `docker pull <dockerhub-user>/video-caption-agent:latest`
- [ ] Confirm image size: `docker images video-caption-agent:latest --format "{{.Size}}"` (target comfortably under the stated limit — this image should land well under 500MB since there are no baked-in model weights)
- [ ] Runtime smoke test: `docker run --rm -e MODEL_PROVIDER=mock -v $(pwd)/sample_input:/input -v /tmp/out:/output video-caption-agent:latest && python3 -m json.tool /tmp/out/results.json`
- [ ] Re-run smoke test with `MODEL_PROVIDER=fireworks` and a real key against 2–3 real clips in the 30s–2min range
- [ ] `results.json` validates with `python -m json.tool` and contains all 4 required styles for every `task_id`
- [ ] GitHub repo is public, README has setup + usage instructions, app is runnable from the instructions alone
- [ ] Double-check Discord/participant guide for any Track-2-specific `ALLOWED_MODELS` restriction before your final scored run
- [ ] Fill in lablab.ai submission fields: title, short/long description, cover image, video presentation, slide presentation, GitHub URL, demo URL

## Known limitations

- Style captions are only as good as the Stage A scene summary; a very fast
  or visually ambiguous clip can produce a thin summary and generic captions.
- The `local` provider (BLIP) is a genuine fallback but noticeably lower
  quality than a real VLM — don't submit your scored run on it.
- No audio/transcript signal is used, only visual frames — if judged clips
  rely on speech/sound for context, consider adding a Whisper-on-Fireworks
  transcript as a third input to Stage A.

## Last-minute improvement ideas

- Add an audio transcript (Fireworks-hosted Whisper) into the Stage A prompt
  for clips where dialogue/sound matters.
- Cache scene summaries by video URL hash so repeated runs against the same
  clip during dev don't re-spend Fireworks tokens.
- If fine-tuning time allows: fine-tune a small open captioner on a
  hand-labeled style dataset per the brief's "fine-tuning is explicitly
  permitted" note, and use it as a second, ensemble-style vote alongside the
  prompted Fireworks output for tone consistency.
