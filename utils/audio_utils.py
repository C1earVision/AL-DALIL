"""
Audio utility functions for loading, resampling, and validating audio files.
"""

import os
import librosa
import numpy as np
import soundfile as sf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUDIO_SAMPLE_RATE, SUPPORTED_AUDIO_FORMATS


def load_audio(file_path: str, target_sr: int = AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load an audio file and resample to the target sample rate.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate (default: 16000 Hz for Whisper).

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio format is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format: '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    # Load and resample using librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

    return audio, sr


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    audio, sr = load_audio(file_path)
    return len(audio) / sr


def validate_audio_file(file_path: str) -> dict:
    """
    Validate an audio file and return its metadata.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary containing:
            - valid (bool): Whether the file is valid
            - duration (float): Duration in seconds
            - sample_rate (int): Original sample rate
            - format (str): File format extension
            - error (str | None): Error message if invalid
    """
    result = {
        "valid": False,
        "duration": 0.0,
        "sample_rate": 0,
        "format": "",
        "error": None,
    }

    try:
        if not os.path.exists(file_path):
            result["error"] = f"File not found: {file_path}"
            return result

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_FORMATS:
            result["error"] = f"Unsupported format: {ext}"
            return result

        result["format"] = ext

        # Get original sample rate without resampling
        info = sf.info(file_path)
        result["sample_rate"] = info.samplerate
        result["duration"] = info.duration
        result["valid"] = True

    except Exception as e:
        result["error"] = f"Error reading audio file: {str(e)}"

    return result


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string (MM:SS or HH:MM:SS).

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    seconds = int(seconds)
    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
