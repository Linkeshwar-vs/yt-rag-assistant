from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str:
    """
    Extract video ID from a YouTube URL.
    """
    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    if parsed.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        return parse_qs(parsed.query)["v"][0]

    raise ValueError("Invalid YouTube URL")

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)