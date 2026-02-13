import os
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# ------------------------------
# 1. Initialise
# ------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"  # adjust filename if different
os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-xxx"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
client = genai.Client(http_options=HttpOptions(api_version="v1"))

# ------------------------------
# 2. Fetch YouTube transcript
# ------------------------------
video_id = "scaOH7hHnEA"  # replace with your video ID
transcript = YouTubeTranscriptApi().fetch(video_id)
transcript_text = " ".join([snippet.text for snippet in transcript])

# ------------------------------
# 3. Create prompt
# ------------------------------
prompt = (
    "Critically evaluate the following YouTube transcript as if it may contain "
    "misinformation or misleading framing. Use up-to-date web sources to fact-check "
    "key claims, but do not limit the analysis to isolated claim verification.\n\n"
    "Identify:\n"
    "- Claims that are false, unsupported, or contradicted by current evidence\n"
    "- Claims that are technically or partially true but presented in a misleading way\n"
    "- Omissions, framing, or rhetorical techniques that distort the broader picture\n\n"
    "Then provide an overall assessment of the video’s central narrative, explaining "
    "whether it fairly represents the evidence or constructs a misleading storyline. "
    "Ground all conclusions in current web information.\n\n"
    + transcript_text
)

# ------------------------------
# 4. Generate
# ------------------------------
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=GenerateContentConfig(
        tools=[
            Tool(
                google_search=GoogleSearch()  # live web search
            )
        ]
    ),
)

# ------------------------------
# 5. Output
# ------------------------------
print(response.text)
