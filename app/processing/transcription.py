import os
import logging
from typing import List, Dict, Any, Tuple

from faster_whisper import WhisperModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class Transcriber:
    """
    Encapsulates the audio transcription logic using the Faster-Whisper model.

    This class is designed to be instantiated once per worker process to avoid
    reloading the expensive AI model into memory on every task.

    It provides a method to transcribe an audio file, intelligently returning both
    a user-friendly, sentence-level transcript and the detailed, word-level
    data required for subsequent diarization mapping.
    """

    def __init__(self, model_path: str = settings.FASTER_WHISPER_MODEL_PATH):
        """
        Initializes the Transcriber and loads the Faster-Whisper model into memory.

        Args:
            model_path (str): The local path to the Faster-Whisper model directory.
        """
        if not model_path or not os.path.isdir(model_path):
            msg = f"Faster-Whisper model path '{model_path}' is invalid or does not exist."
            logger.critical(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading Faster-Whisper model from: {model_path}")
        try:
            # Load the model onto the GPU with float16 precision for optimal performance.
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

        Args:
            audio_path (str): The full path to the audio file to be transcribed.
            language (str): The language of the audio ('vi', 'en', etc.).

        Returns:
            A tuple containing two lists of dictionaries:
            - (sentence_level_transcript, word_level_transcript)
        """
        logger.info(f"Starting transcription for audio file: {audio_path} in language '{language}'")

        try:
            # Use VAD (Voice Activity Detection) to filter out long silences and improve
            # the quality of sentence segmentation.
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                language=language,
                word_timestamps=True,  # This is essential for the diarization mapping step
                vad_filter=True,
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
                            # We don't need a word-specific ID for mapping
                            "word": word.word.strip(),
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                        })

            logger.info(f"Successfully transcribed {audio_path}. "
                        f"Found {len(sentence_level_transcript)} sentences and {len(word_level_transcript)} words.")

            return sentence_level_transcript, word_level_transcript

        except Exception as e:
            logger.error(f"An unexpected error occurred during transcription of {audio_path}: {e}", exc_info=True)
            # Return empty lists in case of failure
            return [], []