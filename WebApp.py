import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# -------------------------
# Default prompt
# -------------------------

DEFAULT_PROMPT = """Critically evaluate the following YouTube transcript as if it may contain misinformation or misleading framing. Use up-to-date web sources to fact-check key claims, but do not limit the analysis to isolated claim verification.

Identify:
- Claims that are false, unsupported, or contradicted by current evidence
- Claims that are technically or partially true but presented in a misleading way
- Omissions, framing, or rhetorical techniques that distort the broader picture

Then provide an overall assessment of the video’s central narrative, explaining whether it fairly represents the evidence or constructs a misleading storyline. Ground all conclusions in current web information.
"""

# -------------------------
# Helpers
# -------------------------

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(snippet.text for snippet in transcript)


def analyze(text_prompt, transcript_text, use_search):

    # --- Vertex / credentials setup ---
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # your JSON file
    os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0584061357"
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
        http_options=HttpOptions(api_version="v1"),
    )

    # --- Prompt assembly ---
    full_prompt = text_prompt + "\n\n" + transcript_text

    config = None
    if use_search:
        config = GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())]
        )

    # --- Model call ---
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=config,
    )

    return response.text


# -------------------------
# UI
# -------------------------

st.title("YouTube Narrative Fact Checker")

video_url = st.text_input("YouTube URL")

use_search = st.checkbox("Enable live web fact-checking", value=True)

with st.expander("Edit analysis prompt"):
    user_prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        height=220,
    )

run = st.button("Analyze")

# -------------------------
# Execution
# -------------------------

if run:

    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    with st.spinner("Fetching transcript..."):
        transcript_text = fetch_transcript(video_id)

    with st.spinner("Analyzing..."):
        result = analyze(user_prompt, transcript_text, use_search)

    st.subheader("Analysis")
    st.write(result)