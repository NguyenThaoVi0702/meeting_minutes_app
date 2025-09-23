import os
import logging
from typing import List, Dict, Any, Tuple

from faster_whisper import WhisperModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class Transcriber:

    def __init__(self, model_path: str = settings.FASTER_WHISPER_MODEL_PATH):
        """
        Initializes the Transcriber and loads the Faster-Whisper model into memory.
        """
        if not model_path or not os.path.isdir(model_path):
            msg = f"Faster-Whisper model path '{model_path}' is invalid or does not exist."
            logger.critical(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading Faster-Whisper model from: {model_path}")
        try:
            self.model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
            logger.info("Faster-Whisper model loaded successfully onto CUDA device.")
        except Exception as e:
            logger.critical(f"FATAL: Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise

    def transcribe(self, audio_path: str, language: str = "vi") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transcribes an audio file and produces two separate transcript formats.

        1.  Sentence-level transcript: Logical chunks of speech suitable for display.
        2.  Word-level transcript: Granular word data with precise timestamps for mapping.
        """
        logger.info(f"Starting transcription for audio file: {audio_path} in language '{language}'")

        try:

            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                word_timestamps=True,  
                vad_filter=False,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            sentence_level_transcript = []
            word_level_transcript = []
            segment_id_counter = 0

            for segment in segments:
                # --- 1. Build the Sentence-Level Transcript (for the user) ---
                sentence_level_transcript.append({
                    "id": segment_id_counter,
                    "text": segment.text.strip(),
                    "start_time": round(segment.start, 3),
                    "end_time": round(segment.end, 3),
                })
                segment_id_counter += 1

                # --- 2. Build the Word-Level Transcript (for the database/diarization) ---
                if segment.words:
                    for word in segment.words:
                        word_level_transcript.append({
                            "word": word.word.strip(),
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                        })

            logger.info(f"Successfully transcribed {audio_path}. "
                        f"Found {len(sentence_level_transcript)} sentences and {len(word_level_transcript)} words.")

            return sentence_level_transcript, word_level_transcript

        except Exception as e:
            logger.error(f"An unexpected error occurred during transcription of {audio_path}: {e}", exc_info=True)
            return [], []
