# YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about YouTube videos using their transcripts.

The project extracts a video's transcript, stores it in a FAISS vector database, retrieves the most relevant sections using MMR retrieval and Cross-Encoder re-ranking, and generates responses with Llama 3.1.

## Features

* Accepts full YouTube URLs
* Extracts video transcripts
* Semantic search using FAISS
* MMR-based retrieval
* Cross-Encoder re-ranking
* Answers using Llama 3.1

## Tech Stack

* Python
* LangChain
* Hugging Face
* FAISS
* Sentence Transformers
* YouTube Transcript API

## Pipeline

```
YouTube URL
      ↓
Transcript Extraction
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS
      ↓
MMR Retrieval
      ↓
Cross-Encoder Re-ranking
      ↓
Llama 3.1
      ↓
Answer
```

## Project Structure

```
app.py
config.py
rag.py
transcript.py
utils.py
```

## Installation

```bash
git clone https://github.com/Linkeshwar-vs/yt-rag-assistant.git
cd yt-rag-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```
HF_TOKEN=your_huggingface_token
```

Run the application:

```bash
python app.py
```

## License

MIT License
