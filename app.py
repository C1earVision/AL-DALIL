try:
    import spaces
except ImportError:
    pass

import gradio as gr

from modules.asr import ArabicASR
from modules.summarizer import ArabicSummarizer
from modules.search import SemanticSearch
from utils.audio_utils import validate_audio_file, format_duration
from config import (
    ASR_MODEL_NAME,
    SUMMARIZATION_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    SUMMARIZATION_MAX_TOKENS,
)

asr = ArabicASR()
summarizer = ArabicSummarizer()
search_engine = SemanticSearch()

app_state = {
    "transcription": "",
    "is_indexed": False,
}

def transcribe_audio(audio_file):
    """Handle audio transcription from the UI."""
    if audio_file is None:
        return (
            "Please upload an audio file.",
            "",
            "",
        )

    audio_path = audio_file if isinstance(audio_file, str) else audio_file

    validation = validate_audio_file(audio_path)
    if not validation["valid"]:
        return (
            f"Invalid audio file: {validation['error']}",
            "",
            "",
        )

    duration_str = format_duration(validation["duration"])

    try:
        result = asr.transcribe_with_segments(audio_path)

        app_state["transcription"] = result["text"]
        app_state["is_indexed"] = False

        segments_display = ""
        for seg in result.get("segments", []):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            segments_display += f"[{start:.1f}s → {end:.1f}s]  {seg['text']}\n\n"

        if not segments_display:
            segments_display = result["text"]

        status = (
            f"Transcription complete!\n"
            f"Audio duration: {duration_str}\n"
            f"Processing time: {result['processing_time']}s\n"
            f"Characters: {len(result['text'])}\n"
            f"Segments: {len(result.get('segments', []))}"
        )

        return status, result["text"], segments_display

    except Exception as e:
        return f"Transcription failed: {str(e)}", "", ""

def summarize_text(text, max_tokens):
    if not text or not text.strip():
        return "No text to summarize. Please transcribe audio first or paste text.", ""

    try:
        result = summarizer.summarize(
            text,
            max_tokens=int(max_tokens),
        )

        status = (
            f"Summarization complete!\n"
            f"Original: {result['original_length']} characters\n"
            f"Summary: {result['summary_length']} characters\n"
            f"Compression: {result['compression_ratio']:.1%}\n"
            f"Processing time: {result['processing_time']}s"
        )

        return status, result["summary"]

    except Exception as e:
        return f"Summarization failed: {str(e)}", ""


def load_transcription_for_summary():
    return app_state.get("transcription", "")


def index_text_for_search(text):
    if not text or not text.strip():
        return "No text to index. Please transcribe audio first or paste text."

    try:
        result = search_engine.index_text(text)
        app_state["is_indexed"] = True

        return (
            f"Text indexed successfully!\n"
            f"Chunks created: {result['num_chunks']}\n"
            f"Embedding dimension: {result['embedding_dim']}\n"
            f"Processing time: {result['processing_time']}s"
        )

    except Exception as e:
        return f"Indexing failed: {str(e)}"


def search_text(query, top_k):
    if not query or not query.strip():
        return "Please enter a search query."

    if not app_state.get("is_indexed", False):
        return "No text has been indexed yet. Please index text first."

    try:
        results = search_engine.search(query, top_k=int(top_k))

        if not results:
            return "No results found."

        # Format results for display
        output = f"Found {len(results)} results for: \"{query}\"\n"
        output += "═" * 60 + "\n\n"

        for r in results:
            score_bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
            output += f"Rank #{r['rank']} — Similarity: {r['score']:.4f}\n"
            output += f"   [{score_bar}]\n"
            output += f"   {r['text']}\n\n"
            output += "─" * 40 + "\n\n"

        return output

    except Exception as e:
        return f"Search failed: {str(e)}"


def load_transcription_for_search():
    return app_state.get("transcription", "")


def create_app():
    custom_css = """
    .arabic-text textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
        font-size: 16px;
        line-height: 1.8;
    }
    .arabic-text-output {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
        font-size: 16px;
        line-height: 1.8;
    }
    .status-box textarea {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
    }
    .gradio-container {
        max-width: 1200px !important;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    .model-info {
        font-size: 12px;
        color: #7f8c8d;
        text-align: center;
    }
    """

    with gr.Blocks(
        title="Al Dalil - Arabic Audio Understanding System",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        ),
    ) as app:

        # Header
        gr.Markdown(
            """
            # 🎧 Al Dalil (الدليل)
            ### Arabic Speech Recognition · Text Summarization · Semantic Search
            """
        )

        gr.Markdown(
            f"""
            <div class="model-info">
            <b>Models:</b> ASR: {ASR_MODEL_NAME} (Groq) | Summarization: {SUMMARIZATION_MODEL_NAME} (Groq) | Embeddings: {EMBEDDING_MODEL_NAME}
            </div>
            """,
        )

        # ── Tab 1: Transcription ──
        with gr.Tab("Transcribe", id="tab-transcribe"):
            gr.Markdown("### Upload Arabic audio to transcribe it to text")

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    transcribe_btn = gr.Button(
                        "🎙️ Transcribe",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=2):
                    transcribe_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["status-box"],
                        lines=5,
                    )

            with gr.Row():
                with gr.Column():
                    transcription_output = gr.Textbox(
                        label="📝 Full Transcription",
                        interactive=True,
                        elem_classes=["arabic-text"],
                        lines=8,
                        rtl=True,
                    )
                with gr.Column():
                    segments_output = gr.Textbox(
                        label="⏱️ Timestamped Segments",
                        interactive=False,
                        elem_classes=["arabic-text"],
                        lines=8,
                        rtl=True,
                    )

            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[audio_input],
                outputs=[transcribe_status, transcription_output, segments_output],
            )

        with gr.Tab("Summarize", id="tab-summarize"):
            gr.Markdown("### Summarize Arabic text from transcription or manual input")

            with gr.Row():
                with gr.Column(scale=1):
                    load_for_summary_btn = gr.Button(
                        "📋 Load Transcription",
                        variant="secondary",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=2048,
                        value=SUMMARIZATION_MAX_TOKENS,
                        step=50,
                        label="Max Summary Length (tokens)",
                    )
                    summarize_btn = gr.Button(
                        "📝 Summarize",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=2):
                    summary_input = gr.Textbox(
                        label="📄 Input Text",
                        lines=6,
                        rtl=True,
                        elem_classes=["arabic-text"],
                        placeholder="Paste Arabic text here or load from transcription...",
                    )
                    summary_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["status-box"],
                        lines=4,
                    )
                    summary_output = gr.Textbox(
                        label="📋 Summary",
                        interactive=False,
                        rtl=True,
                        elem_classes=["arabic-text"],
                        lines=4,
                    )

            load_for_summary_btn.click(
                fn=load_transcription_for_summary,
                inputs=[],
                outputs=[summary_input],
            )

            summarize_btn.click(
                fn=summarize_text,
                inputs=[summary_input, max_tokens_slider],
                outputs=[summary_status, summary_output],
            )

        with gr.Tab("Search", id="tab-search"):
            gr.Markdown("### Search within the transcribed text using natural language queries")

            with gr.Row():
                with gr.Column(scale=1):
                    load_for_search_btn = gr.Button(
                        "📋 Load Transcription",
                        variant="secondary",
                    )
                    search_text_input = gr.Textbox(
                        label="📄 Text to Index",
                        lines=5,
                        rtl=True,
                        elem_classes=["arabic-text"],
                        placeholder="Paste text here or load from transcription...",
                    )
                    index_btn = gr.Button(
                        "📊 Build Search Index",
                        variant="secondary",
                    )
                    index_status = gr.Textbox(
                        label="Index Status",
                        interactive=False,
                        elem_classes=["status-box"],
                        lines=4,
                    )

                with gr.Column(scale=2):
                    search_query = gr.Textbox(
                        label="🔍 Search Query",
                        lines=2,
                        rtl=True,
                        elem_classes=["arabic-text"],
                        placeholder="Enter your search query in Arabic...",
                    )
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Number of Results",
                        )
                        search_btn = gr.Button(
                            "🔍 Search",
                            variant="primary",
                            size="lg",
                        )
                    search_results = gr.Textbox(
                        label="📊 Search Results",
                        interactive=False,
                        lines=15,
                        elem_classes=["arabic-text"],
                    )

            load_for_search_btn.click(
                fn=load_transcription_for_search,
                inputs=[],
                outputs=[search_text_input],
            )

            index_btn.click(
                fn=index_text_for_search,
                inputs=[search_text_input],
                outputs=[index_status],
            )

            search_btn.click(
                fn=search_text,
                inputs=[search_query, top_k_slider],
                outputs=[search_results],
            )

        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #95a5a6; font-size: 12px;">
            Al Dalil — NLP Project 2<br>
            Pipeline: Audio → Whisper ASR (Groq) → Text → LLM Summarization (Groq) → Embeddings → FAISS Search
            </div>
            """
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
