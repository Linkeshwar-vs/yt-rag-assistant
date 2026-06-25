from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from utils import extract_video_id


def load_transcript(video_url: str) -> str:

    video_id = extract_video_id(video_url)

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        transcript_text = " ".join(
            snippet.text for snippet in transcript
        )

        return transcript_text

    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")

    except Exception as e:
        raise Exception(f"Transcript Error: {e}")