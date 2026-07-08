import json
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from captioning import REQUIRED_STYLES, generate_captions_for_task
from model_client import build_client, encode_jpeg_files
from video_utils import cleanup_dir, download_video, extract_frames, make_scratch_dir

st.set_page_config(
    page_title="Grounded Video Captioning Agent",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Grounded Multi-Style Video Captioning Agent")
st.caption("AMD Developer Hackathon ACT II — Track 2")

st.markdown(
    """
This demo generates grounded captions for a video in four styles:
**formal**, **sarcastic**, **humorous_tech**, and **humorous_non_tech**.
"""
)

video_url = st.text_input(
    "Video URL",
    value="https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
)

max_frames = st.slider("Frames to sample", 1, 6, 3)

if st.button("Generate Captions", type="primary"):
    if not video_url.strip():
        st.error("Please enter a video URL.")
        st.stop()

    os.environ.setdefault("MODEL_PROVIDER", "fireworks")
    os.environ.setdefault("MODEL_NAME", "accounts/fireworks/models/kimi-k2p6")
    os.environ["MAX_FRAMES"] = str(max_frames)
    os.environ.setdefault("FRAME_MAX_SIDE", "768")
    os.environ.setdefault("JPEG_QUALITY", "5")
    os.environ.setdefault("MODEL_TIMEOUT_SECONDS", "180")

    scratch = make_scratch_dir()

    try:
        with st.status("Processing video...", expanded=True) as status:
            st.write("Downloading video...")
            video_path = download_video(video_url, scratch)

            st.write("Sampling frames...")
            frame_paths = extract_frames(video_path, scratch, max_frames_override=max_frames)

            st.write("Encoding frames...")
            images_b64 = encode_jpeg_files(frame_paths)

            st.write("Calling vision model and generating captions...")
            client = build_client()
            captions = generate_captions_for_task(client, images_b64, REQUIRED_STYLES)

            status.update(label="Done", state="complete")

        st.subheader("Generated Captions")

        for style, caption in captions.items():
            st.markdown(f"### `{style}`")
            st.write(caption)

        st.subheader("Output JSON")
        result = {
            "task_id": "demo_video",
            "captions": captions,
        }
        st.json(result)

    except Exception as exc:
        st.error(f"Demo failed: {exc}")

    finally:
        cleanup_dir(scratch)
        try:
            os.rmdir(scratch)
        except OSError:
            pass