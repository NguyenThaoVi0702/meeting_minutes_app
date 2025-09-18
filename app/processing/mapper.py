import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def map_speaker_to_text(
    diarization_output: List[Dict[str, Any]],
    word_level_transcript: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assigns speaker labels to a word-level transcript based on a diarization timeline.
    """
    # --- Input Validation ---
    if not diarization_output or not word_level_transcript:
        logger.warning("Mapping received empty diarization or word transcript data. Returning an empty list.")
        return []

    logger.info(f"Starting mapping of {len(diarization_output)} speaker segments to {len(word_level_transcript)} words.")

    final_diarized_transcript = []
    word_iterator_idx = 0  

    for speaker_segment in diarization_output:
        segment_start = speaker_segment['start_s']
        segment_end = speaker_segment['end_s']
        speaker_label = speaker_segment['speaker']

        segment_words = []

        while word_iterator_idx < len(word_level_transcript):
            word_data = word_level_transcript[word_iterator_idx]
            word_start = word_data['start']
            word_end = word_data['end']

            word_center = word_start + (word_end - word_start) / 2

            if word_center < segment_start:
                word_iterator_idx += 1
                continue

            if word_center <= segment_end:
                segment_words.append(word_data['word'])
                word_iterator_idx += 1

            else:
                break

        if segment_words:
            full_text = " ".join(segment_words)
            final_diarized_transcript.append({
                "speaker": speaker_label,
                "text": full_text,
                "start_time": segment_start,
                "end_time": segment_end,
            })

    logger.info(f"Successfully mapped and created {len(final_diarized_transcript)} final diarized segments.")
    return final_diarized_transcript
