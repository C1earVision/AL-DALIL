---
title: Al Dalil
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Live Demo
https://huggingface.co/spaces/Ahmed32323/Al-Dalil

# 🎧 Al Dalil (الدليل)

**Al Dalil** is a deep learning-based system for Arabic audio understanding that combines **Speech Recognition**, **Text Summarization**, and **Semantic Search** into a unified pipeline.

## 📋 Pipeline

```
Audio → Whisper ASR → Arabic Text → mT5 Summarization → Embeddings → FAISS Search
```

## 🧠 Models Used

| Task | Model | Description |
|---|---|---|
| **Speech-to-Text** | `openai/whisper-large-v3` | State-of-the-art multilingual ASR with excellent Arabic support |
| **Summarization** | `csebuetnlp/mT5-small` | Multilingual T5 model for text summarization |
| **Semantic Search** | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual sentence embeddings (384-dim) |
| **Search Index** | FAISS (CPU) | Fast approximate nearest neighbor search |

All models are **pretrained** and loaded from Hugging Face — no training from scratch required.

## 🚀 Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (required for audio processing)

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **Mac**: `brew install ffmpeg`

## 🖥️ Usage

### Web Interface (Gradio)

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

The web interface provides 3 tabs:
1. **🎙️ Transcribe**: Upload audio files and get Arabic transcriptions with timestamps
2. **📝 Summarize**: Generate summaries of the transcribed text
3. **🔍 Search**: Search within the transcribed content using natural language queries

### Python API

```python
from modules.asr import ArabicASR
from modules.summarizer import ArabicSummarizer
from modules.search import SemanticSearch

# 1. Transcribe audio
asr = ArabicASR()
result = asr.transcribe("audio.wav")
print(result["text"])

# 2. Summarize text
summarizer = ArabicSummarizer()
summary = summarizer.summarize(result["text"])
print(summary["summary"])

# 3. Semantic search
search = SemanticSearch()
search.index_text(result["text"])
results = search.search("search query in Arabic", top_k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")
```

## 📁 Project Structure

```
Project 2/
├── app.py                  # Gradio web interface (main entry point)
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── modules/
│   ├── __init__.py
│   ├── asr.py              # Speech-to-Text (Whisper large-v3)
│   ├── summarizer.py       # Text Summarization (mT5)
│   └── search.py           # Semantic Search (sentence-transformers + FAISS)
├── utils/
│   ├── __init__.py
│   └── audio_utils.py      # Audio loading/validation utilities
└── data/
    └── (audio files)
```

## ⚙️ Configuration

This project requires a `config.py` file to store settings and API keys. This file is not included in the repository for security reasons.

### 1. Create `config.py`
Create a new file named `config.py` in the root directory of the project.

### 2. Add Configuration Variables
Copy and paste the following template into your `config.py` file. **Make sure to replace `"your_groq_api_key_here"` with your actual Groq API key.**

```python
GROQ_API_KEY = "your_groq_api_key_here"

# ASR (Speech-to-Text) Configuration
ASR_MODEL_NAME = "whisper-large-v3"
ASR_LANGUAGE = "ar"
AUDIO_SAMPLE_RATE = 16000

# Summarization Configuration
SUMMARIZATION_MODEL_NAME = "llama-3.3-70b-versatile" 
SUMMARIZATION_MAX_TOKENS = 1024
SUMMARIZATION_TEMPERATURE = 0.3

# Semantic Search Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
SEARCH_TOP_K = 5
SEARCH_CHUNK_SIZE = 2 

# Supported Audio Formats
SUPPORTED_AUDIO_FORMATS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")
```

## 📝 Notes

- **GPU Recommended**: Whisper large-v3 benefits greatly from a CUDA GPU (16GB+ VRAM recommended)
- **First Run**: Models are downloaded automatically from Hugging Face on first use (~3GB for Whisper large-v3)
- **Audio Formats**: Supports `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.webm`
- **Audio Resampling**: All audio is automatically resampled to 16kHz
