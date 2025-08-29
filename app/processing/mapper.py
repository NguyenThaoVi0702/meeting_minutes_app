import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def map_speaker_to_text(
    diarization_output: List[Dict[str, Any]],
    word_level_transcript: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assigns speaker labels to a word-level transcript based on a diarization timeline.

    This function efficiently maps the output of the transcription model to the output
    of the diarization model. It iterates through both chronological lists a single
    time, making it highly performant for long meetings.

    The core logic assigns a word to a speaker segment if the word's center
    timestamp falls within the speaker segment's start and end times.

    Args:
        diarization_output: A list of speaker segments, each with 'start_s',
                            'end_s', and 'speaker' keys.
                            Example: [{'start_s': 0.5, 'end_s': 4.2, 'speaker': 'LinhPT'}]

        word_level_transcript: A list of words, each with 'word', 'start', and
                               'end' keys.
                               Example: [{'word': 'Hello', 'start': 0.6, 'end': 1.1}]

    Returns:
        A list of dictionaries representing the final, speaker-separated transcript,
        matching the 'DiarizedSegment' schema.
        Example: [{'speaker': 'LinhPT', 'text': 'Hello world', 'start_time': 0.5, 'end_time': 4.2}]
    """
    # --- Input Validation ---
    if not diarization_output or not word_level_transcript:
        logger.warning("Mapping received empty diarization or word transcript data. Returning an empty list.")
        return []

    logger.info(f"Starting mapping of {len(diarization_output)} speaker segments to {len(word_level_transcript)} words.")

    final_diarized_transcript = []
    word_iterator_idx = 0  # A pointer to avoid re-scanning the word list

    # --- Main Mapping Loop ---
    # Iterate through each identified speaker turn from the diarization process
    for speaker_segment in diarization_output:
        segment_start = speaker_segment['start_s']
        segment_end = speaker_segment['end_s']
        speaker_label = speaker_segment['speaker']

        segment_words = []

        # Efficiently scan the word list starting from where we left off
        while word_iterator_idx < len(word_level_transcript):
            word_data = word_level_transcript[word_iterator_idx]
            word_start = word_data['start']
            word_end = word_data['end']

            # Determine the word's center time for robust assignment
            word_center = word_start + (word_end - word_start) / 2

            # If the word's center is before the current speaker segment, it might be
            # in a silent gap. We skip it and check the next word for this same segment.
            if word_center < segment_start:
                word_iterator_idx += 1
                continue

            # If the word's center is within the current speaker segment, assign it.
            if word_center <= segment_end:
                segment_words.append(word_data['word'])
                word_iterator_idx += 1
            # If the word's center is past the end of this segment, it belongs to
            # a future segment. We break the inner loop and move to the next speaker segment.
            else:
                break

        # --- Assemble Segment ---
        # If any words were assigned to this speaker segment, join them into a
        # single string and create the final transcript object.
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