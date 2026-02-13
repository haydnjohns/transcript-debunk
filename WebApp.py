import os
import json
import re
import tempfile
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# ------------------------------------------------
# Load Vertex credentials from Streamlit secrets
# ------------------------------------------------

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:

    creds = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    json.dump(creds, open(temp_file.name, "w"))

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
    os.environ["GOOGLE_CLOUD_PROJECT"] = st.secrets["GOOGLE_CLOUD_PROJECT"]
    os.environ["GOOGLE_CLOUD_LOCATION"] = st.secrets["GOOGLE_CLOUD_LOCATION"]
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = st.secrets["GOOGLE_GENAI_USE_VERTEXAI"]

# ------------------------------------------------
# Default prompt
# ------------------------------------------------

DEFAULT_PROMPT = """Critically evaluate the following YouTube transcript as if it may contain misinformation or misleading framing. Use up-to-date web sources to fact-check key claims, but do not limit the analysis to isolated claim verification.

Identify:
- Claims that are false, unsupported, or contradicted by current evidence
- Claims that are technically or partially true but presented in a misleading way
- Omissions, framing, or rhetorical techniques that distort the broader picture

Then provide an overall assessment of the video’s central narrative, explaining whether it fairly represents the evidence or constructs a misleading storyline. Ground all conclusions in current web information.
"""

# ------------------------------------------------
# Helpers
# ------------------------------------------------

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


def analyze(prompt_text, transcript_text, use_search):

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
        http_options=HttpOptions(api_version="v1"),
    )

    full_prompt = prompt_text + "\n\n" + transcript_text

    config = None
    if use_search:
        config = GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())]
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_prompt,
        config=config,
    )

    return response.text


# ------------------------------------------------
# UI
# ------------------------------------------------

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

# ------------------------------------------------
# Execution
# ------------------------------------------------

if run:

    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    with st.spinner("Fetching transcript..."):
        transcript_text = fetch_transcript(video_id)

    with st.spinner("Analyzing video narrative..."):
        result = analyze(user_prompt, transcript_text, use_search)

    st.subheader("Analysis")
    st.write(result)