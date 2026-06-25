from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from utils import extract_video_id

def load_transcript(video_url: str) -> str:

    video_id = extract_video_id(video_url)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en"],
        )

        transcript = " ".join(
            chunk["text"] for chunk in transcript_list
        )

        return transcript

    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")

    except Exception as e:
        raise Exception(f"Transcript Error : {e}")