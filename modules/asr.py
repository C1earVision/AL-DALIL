import time
import os
import sys

from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, ASR_MODEL_NAME, ASR_LANGUAGE


class ArabicASR:

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = ASR_MODEL_NAME

    def transcribe(self, audio_path: str) -> dict:
        print(f"[ASR] Transcribing via Groq API: {audio_path}")
        start = time.time()

        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=self.model_name,
                language=ASR_LANGUAGE,
                response_format="verbose_json",
            )

        elapsed = time.time() - start
        print(f"[ASR] Transcription completed in {elapsed:.1f}s")

        return {
            "text": transcription.text.strip(),
            "segments": getattr(transcription, "segments", []),
            "processing_time": round(elapsed, 2),
            "model": self.model_name,
        }

    def transcribe_with_segments(self, audio_path: str) -> dict:
        result = self.transcribe(audio_path)

        segments = []
        raw_segments = result.get("segments", [])

        if raw_segments:
            for i, seg in enumerate(raw_segments):
                if hasattr(seg, "start"):
                    segments.append({
                        "id": i + 1,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                    })
                elif isinstance(seg, dict):
                    segments.append({
                        "id": i + 1,
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip(),
                    })

        return {
            "text": result["text"],
            "segments": segments,
            "processing_time": result["processing_time"],
        }
