import time
import os
import sys

from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GROQ_API_KEY,
    SUMMARIZATION_MODEL_NAME,
    SUMMARIZATION_MAX_TOKENS,
    SUMMARIZATION_TEMPERATURE,
)


class ArabicSummarizer:
    def __init__(self):
        """Initialize the Groq client."""
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = SUMMARIZATION_MODEL_NAME

    def summarize(
        self,
        text: str,
        max_tokens: int = None,
    ) -> dict:
        if max_tokens is None:
            max_tokens = SUMMARIZATION_MAX_TOKENS

        if not text or not text.strip():
            return {
                "summary": "",
                "original_length": 0,
                "summary_length": 0,
                "compression_ratio": 0.0,
                "processing_time": 0.0,
            }

        print(f"[Summarizer] Summarizing text ({len(text)} characters) via Groq API...")
        start = time.time()

        system_prompt = (
            "أنت مساعد متخصص في تلخيص النصوص العربية. "
            "قم بتلخيص النص التالي بشكل موجز ودقيق باللغة العربية. "
            "حافظ على المعلومات الأساسية والأفكار الرئيسية. "
            "اكتب الملخص باللغة العربية فقط."
        )

        user_prompt = f"قم بتلخيص النص التالي:\n\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=SUMMARIZATION_TEMPERATURE,
            )

            summary = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[Summarizer] Error: {e}")
            summary = f"Error generating summary: {str(e)}"

        elapsed = time.time() - start
        print(f"[Summarizer] Summarization completed in {elapsed:.1f}s")

        original_length = len(text)
        summary_length = len(summary)

        return {
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": round(summary_length / original_length, 3) if original_length > 0 else 0,
            "processing_time": round(elapsed, 2),
        }
