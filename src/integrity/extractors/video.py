"""Video/audio extractor using ffprobe, ffmpeg, and faster-whisper."""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional
from src.integrity.models import ExtractionResult, ExtractionStatus, Segment, Modality


def _get_ffmpeg_cmd(cmd_name: str) -> str:
    """Find ffmpeg or ffprobe executable in PATH or standard winget location."""
    # Check if directly in PATH
    found = shutil.which(cmd_name)
    if found:
        return found

    # Check common Windows winget installation paths
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages"),
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ffmpeg\bin",
    ]
    for candidate_dir in candidates:
        if os.path.exists(candidate_dir):
            for root, _, files in os.walk(candidate_dir):
                if f"{cmd_name}.exe".lower() in [f.lower() for f in files]:
                    return os.path.join(root, f"{cmd_name}.exe")

    return cmd_name


def check_audio_stream(file_path: str) -> bool:
    """Check if the media file contains at least one audio stream using ffprobe."""
    ffprobe_bin = _get_ffmpeg_cmd("ffprobe")
    try:
        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            file_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        return False


def extract_audio_track(video_path: str, output_wav_path: str) -> bool:
    """Extract audio track as 16kHz mono WAV suitable for Whisper."""
    ffmpeg_bin = _get_ffmpeg_cmd("ffmpeg")
    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            output_wav_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
    except Exception:
        return False


def extract_video(
    file_path: str,
    doc_id: str,
    whisper_model: str = "tiny",
    compute_type: str = "int8",
) -> ExtractionResult:
    """Extract and transcribe speech from an MP4 video file.
    
    Robustness guarantees:
    - Verifies file existence and non-zero size.
    - Inspects streams with ffprobe; returns WARNING for silent videos (no audio track).
    - Extracts 16kHz mono audio into a temp directory and reliably cleans it up.
    - Uses faster-whisper with VAD filtering to accurately timestamp speech segments.
    - Detects empty transcripts / non-speech audio (returns WARNING).
    - Never raises unhandled exceptions.
    """
    if not os.path.exists(file_path):
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=["File does not exist"],
        )

    if os.path.getsize(file_path) == 0:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            status=ExtractionStatus.FAILED,
            warnings=["zero-byte file"],
        )

    # Check for presence of an audio stream
    has_audio = check_audio_stream(file_path)
    if not has_audio:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            segments=[],
            status=ExtractionStatus.WARNING,
            warnings=["no audio track"],
        )

    temp_wav = None
    try:
        # Create temp file for extracted audio
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"audio_{doc_id}_{os.getpid()}.wav")

        success = extract_audio_track(file_path, temp_wav)
        if not success:
            return ExtractionResult(
                doc_id=doc_id,
                file_path=file_path,
                segments=[],
                status=ExtractionStatus.WARNING,
                warnings=["failed to extract audio track via ffmpeg"],
            )

        # Transcribe using faster-whisper.
        # Note: PyAV 14.0+ removed the 'metadata_errors' parameter from av.open(), which causes
        # faster-whisper to crash when opening video/audio files directly in newer environments.
        # We work around this by reading 16kHz mono WAV frames via the standard library 'wave'
        # module, converting them to float32 numpy array, and passing that directly to transcribe().
        try:
            import wave
            import numpy as np
            from faster_whisper import WhisperModel

            with wave.open(temp_wav, "rb") as wf:
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            model = WhisperModel(whisper_model, device="cpu", compute_type=compute_type)
            segments_gen, _ = model.transcribe(audio_np, vad_filter=True)
            whisper_segments = list(segments_gen)
        except ImportError:
            # Fallback if faster-whisper is not yet installed or available
            return ExtractionResult(
                doc_id=doc_id,
                file_path=file_path,
                segments=[],
                status=ExtractionStatus.WARNING,
                warnings=["faster-whisper is not installed or available"],
            )

        segments: List[Segment] = []
        total_words = 0

        for seg in whisper_segments:
            text = seg.text.strip()
            if not text:
                continue
            words = text.split()
            total_words += len(words)
            segments.append(
                Segment(
                    text=text,
                    doc_id=doc_id,
                    modality=Modality.VIDEO.value,
                    start_sec=round(seg.start, 2),
                    end_sec=round(seg.end, 2),
                )
            )

        if total_words < 5:
            return ExtractionResult(
                doc_id=doc_id,
                file_path=file_path,
                segments=[],
                status=ExtractionStatus.WARNING,
                warnings=["no speech detected"],
            )

        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            segments=segments,
            status=ExtractionStatus.OK,
        )

    except Exception as exc:
        return ExtractionResult(
            doc_id=doc_id,
            file_path=file_path,
            segments=[],
            status=ExtractionStatus.WARNING,
            warnings=[f"Audio processing error: {str(exc)}"],
        )
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
