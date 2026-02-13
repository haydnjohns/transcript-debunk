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

DEFAULT_PROMPT = """You are an expert fact‑checker. Review this transcript for misinformation and misleading framing. Produce **short, punchy notes** in bullet points only.

For each claim in the transcript that is found to be less than true:

- **Label the claim** briefly
- State whether it is **Partially True / False / Unsupported / True but misleading**
- Provide **a single short sentence** explaining why, citing current web facts

Then give a **brief overall summary (3–5 bullets)** evaluating the overarching narrative, focusing on whether the video’s story is misleading.
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


def analyse(prompt_text, transcript_text, use_search):

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
st.info("Disclaimer: This tool uses AI which can make mistakes. Check important outputs.")
st.set_page_config(
    page_title="YouTube Debunker",  # Tab title
)
video_url = st.text_input("YouTube URL")
label = "Enable live web fact-checking"
help = "Toggle on only for current events which are likely beyond LLM training data."
use_search = st.checkbox(label=label, help=help,value=False)

with st.expander("Edit analysis prompt"):
    user_prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        height=300,
    )

run = st.button("Analyse")

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

    with st.spinner("Analysing video narrative..."):
        result = analyse(user_prompt, transcript_text, use_search)

    st.subheader("Analysis")
    st.write(result)