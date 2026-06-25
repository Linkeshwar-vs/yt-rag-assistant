# YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about any YouTube video using its transcript.

Instead of relying on general knowledge, the assistant retrieves relevant transcript chunks using semantic search and generates context-aware answers using **Llama 3.1**.


## Features

- Supports full YouTube URLs
- Extracts transcripts directly from YouTube videos
- Splits transcripts into semantic chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in a FAISS vector database
- Retrieves the most relevant transcript sections
- Answers questions using Llama 3.1 via HuggingFace


## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| LLM | Llama 3.1 8B Instruct |
| Framework | LangChain |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Transcript Source | YouTube Transcript API |
| Environment | Python Dotenv |


## Project Structure

```
youtube-rag-assistant/
│
├── app.py              # Entry point
├── config.py           # Configuration values
├── rag.py              # RAG pipeline
├── transcript.py       # Transcript extraction
├── utils.py            # Helper functions
│
├── requirements.txt
├── README.md
└── .gitignore
```



## Architecture

```
                User
                  │
                  ▼
         Enter YouTube URL
                  │
                  ▼
      YouTube Transcript API
                  │
                  ▼
      Transcript Extraction
                  │
                  ▼
     Recursive Text Splitter
                  │
                  ▼
SentenceTransformer Embeddings
                  │
                  ▼
          FAISS Vector Store
                  │
                  ▼
            Similarity Search
                  │
                  ▼
             Prompt Template
                  │
                  ▼
        Llama 3.1 (HuggingFace)
                  │
                  ▼
            Generated Answer
```


## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/youtube-rag-assistant.git

cd youtube-rag-assistant
```

Install dependencies

```bash
pip install -r requirements.txt
```



## Environment Variables

Create a `.env` file in the project root.

```
HF_TOKEN=your_huggingface_token
```

You can obtain your Hugging Face access token from:

https://huggingface.co/settings/tokens



## Author

**Linkeshwar VS**

B.Tech Computer Science and Engineering (AI & ML)

VIT Chennai