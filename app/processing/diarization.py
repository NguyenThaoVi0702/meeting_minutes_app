import os
import time
import logging
import tempfile
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from typing import Optional, List, Dict, Any

from app.core.config import settings
from app.processing.enrollment import SpeakerEnrollment  # Used for type hinting

logger = logging.getLogger(__name__)

class SpeakerDiarization:
    """
    Performs speaker diarization on a given audio file.

    This class identifies speech segments, extracts speaker embeddings from them,
    compares them against a provided list of known speakers, and clusters any
    remaining unknown speakers. It is designed to be run in a background worker.
    """

    def __init__(self):
        """
        Initializes the SpeakerDiarization service and its parameters from settings.
        """
        # Load diarization parameters from the central configuration
        self.segment_duration_s = settings.DIAR_SEG_DURATION
        self.segment_overlap_s = settings.DIAR_SEG_OVERLAP
        self.known_speaker_similarity_threshold = settings.DIAR_KNOWN_THRESH
        self.hac_distance_threshold = settings.DIAR_HAC_THRESH
        self.merge_max_pause_s = settings.DIAR_MERGE_PAUSE
        self.vad_threshold = settings.DIAR_VAD_THRESH
        self.enable_vad = settings.ENABLE_VAD

        # Ensure overlap is not too large
        if self.segment_overlap_s >= self.segment_duration_s:
            logger.warning("Segment overlap is greater than or equal to duration. Adjusting to duration / 3.")
            self.segment_overlap_s = self.segment_duration_s / 3.0

        # Load the VAD model for identifying speech segments
        self.vad_model = None
        try:
            self.vad_model = load_silero_vad()
            logger.info("Silero VAD model loaded successfully for diarization.")
        except Exception as e:
            logger.error(f"Error loading Silero VAD model: {e}. VAD will be disabled.", exc_info=True)

        logger.info("SpeakerDiarization service initialized.")

    def diarize(self, audio_path: str, enrolled_profiles: List[Dict], embedding_service: SpeakerEnrollment) -> List[Dict[str, Any]]:
        """
        Main method to process an audio file and return a diarization timeline.

        Args:
            audio_path (str): The full path to the audio file to process.
            enrolled_profiles (List[Dict]): A list of dictionaries, where each dict
                represents a known speaker profile from the database.
            embedding_service (SpeakerEnrollment): An instance of the embedding service
                used to generate embeddings for the audio segments.

        Returns:
            A list of dictionaries representing the diarization timeline, with speaker labels.
        """
        t_overall_start = time.time()
        logger.info(f"--- Starting Diarization for: {audio_path} ---")

        if not enrolled_profiles:
            logger.warning("No enrolled speaker profiles were provided. All speakers will be labeled as 'Unknown'.")
        else:
            logger.info(f"Diarizing with {len(enrolled_profiles)} known speaker profiles.")

        # Use a temporary directory for intermediate files (like resampled audio)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Preprocess the audio (resample to the target sample rate)
            processed_audio_path = self._preprocess_audio(audio_path, temp_dir_path, embedding_service.target_sr)
            if not processed_audio_path:
                logger.error("Audio preprocessing failed. Aborting diarization.")
                return []

            # Step 1: Apply Voice Activity Detection (VAD) to find speech regions
            speech_timestamps = self._apply_vad(processed_audio_path)

            # Step 2: Segment the speech regions and generate speaker embeddings for each segment
            all_segments_info = self._segment_and_embed(processed_audio_path, speech_timestamps, embedding_service)
            if not all_segments_info:
                logger.error("No audio segments could be embedded. Aborting diarization.")
                return []

            # Step 3: Identify known speakers by comparing segment embeddings to enrolled profiles
            raw_diarization_results, unknown_segments = self._identify_known_speakers(all_segments_info, enrolled_profiles)

            # Step 4: Cluster any remaining unknown speaker segments
            if unknown_segments:
                unknown_speaker_results = self._cluster_unknown_speakers(unknown_segments)
                raw_diarization_results.extend(unknown_speaker_results)

            if not raw_diarization_results:
                logger.warning("No diarization results were generated after all steps.")
                return []

            # Step 5: Merge consecutive segments from the same speaker for a clean timeline
            merged_timeline = self._merge_timeline_segments(raw_diarization_results)

        total_time = time.time() - t_overall_start
        logger.info(f"--- Diarization Complete. Total time: {total_time:.2f}s ---")
        return merged_timeline

    def _preprocess_audio(self, input_path: str, output_dir: Path, target_sr: int) -> Optional[str]:
        """Resamples audio to the target sample rate required by the embedding model."""
        try:
            audio_data, sr_orig = librosa.load(input_path, sr=None, mono=True)

            if sr_orig != target_sr:
                audio_data = librosa.resample(y=audio_data, orig_sr=sr_orig, target_sr=target_sr)

            output_path = output_dir / f"preprocessed_{Path(input_path).stem}.wav"
            sf.write(output_path, audio_data, target_sr)

            return str(output_path)
        except Exception as e:
            logger.error(f"Error preprocessing audio {input_path}: {e}", exc_info=True)
            return None

    def _apply_vad(self, audio_path: str) -> List[Dict[str, float]]:
        """Applies Silero VAD to find speech timestamps."""
        if not self.vad_model:
            logger.warning("VAD model not available, cannot apply VAD.")
            return []
        try:
            wav_tensor = read_audio(audio_path, sampling_rate=16000) # Silero VAD works best at 16k
            speech_ts = get_speech_timestamps(wav_tensor, self.vad_model, return_seconds=True, threshold=self.vad_threshold)
            return speech_ts
        except Exception as e:
            logger.error(f"Error during Silero VAD processing: {e}", exc_info=True)
            return []

    def _segment_and_embed(self, audio_path: str, speech_timestamps: List[Dict], embedding_service: SpeakerEnrollment) -> List[Dict]:
        """Segments audio based on VAD and extracts speaker embeddings."""
        try:
            y_full, sr = librosa.load(audio_path, sr=embedding_service.target_sr)
            segment_len_samples = int(self.segment_duration_s * sr)
            step_samples = int((self.segment_duration_s - self.segment_overlap_s) * sr)

            segments_info = []
            segment_id = 0

            # If VAD failed or was skipped, process the entire file as one region
            process_regions = speech_timestamps if speech_timestamps else [{"start": 0, "end": len(y_full) / sr}]

            for region in process_regions:
                start_sample = int(region["start"] * sr)
                end_sample = int(region["end"] * sr)
                region_audio = y_full[start_sample:end_sample]

                for seg_start in range(0, len(region_audio), step_samples):
                    seg_end = seg_start + segment_len_samples
                    if seg_end > len(region_audio):
                        # Use the last possible full segment instead of a partial one
                        seg_end = len(region_audio)
                        seg_start = seg_end - segment_len_samples
                        if seg_start < 0: continue

                    segment_audio = region_audio[seg_start:seg_end]

                    # Use the passed embedding service to generate the embedding
                    embedding = embedding_service.get_embedding_from_audio(segment_audio)

                    segments_info.append({
                        "id": segment_id,
                        "start_s": (start_sample + seg_start) / sr,
                        "end_s": (start_sample + seg_end) / sr,
                        "embedding": embedding,
                    })
                    segment_id += 1

            logger.info(f"Extracted embeddings for {len(segments_info)} segments.")
            return segments_info
        except Exception as e:
            logger.error(f"Failed during segmenting and embedding stage: {e}", exc_info=True)
            return []

    def _identify_known_speakers(self, all_segments: List[Dict], enrolled_profiles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Compares segment embeddings to known profiles using cosine similarity."""
        known_results = []
        unknown_segments = []

        if not enrolled_profiles:
            return [], all_segments

        for seg in all_segments:
            emb = seg.get("embedding")
            if not isinstance(emb, np.ndarray):
                continue

            best_sim = -1.0
            best_match_payload = None

            for profile in enrolled_profiles:
                # Cosine similarity is 1 - cosine distance
                sim = 1 - cosine(emb, profile["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_match_payload = profile["payload"]

            if best_sim >= self.known_speaker_similarity_threshold:
                speaker_name = best_match_payload.get("display_name", "Known_Speaker_NoName")
                known_results.append({
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "speaker": speaker_name,
                })
            else:
                unknown_segments.append(seg)

        logger.info(f"Identified {len(known_results)} known segments, {len(unknown_segments)} unknown segments remain.")
        return known_results, unknown_segments

    def _cluster_unknown_speakers(self, unknown_segments: List[Dict]) -> List[Dict]:
        """Clusters remaining unknown speaker segments using Hierarchical Agglomerative Clustering (HAC)."""
        if len(unknown_segments) < 2:
            if unknown_segments:
                seg = unknown_segments[0]
                return [{"start_s": seg["start_s"], "end_s": seg["end_s"], "speaker": "Unknown_Speaker_0"}]
            return []

        embeddings = np.array([s["embedding"] for s in unknown_segments])
        try:
            hac = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=self.hac_distance_threshold
            )
            labels = hac.fit_predict(embeddings)

            results = []
            for i, seg in enumerate(unknown_segments):
                results.append({
                    "start_s": seg["start_s"],
                    "end_s": seg["end_s"],
                    "speaker": f"Unknown_Speaker_{labels[i]}"
                })

            logger.info(f"Clustered unknown segments into {len(np.unique(labels))} distinct groups.")
            return results
        except Exception as e:
            logger.error(f"Error during HAC clustering of unknown speakers: {e}", exc_info=True)
            return []

    def _merge_timeline_segments(self, timeline: List[Dict]) -> List[Dict]:
        """Merges consecutive segments from the same speaker for a cleaner timeline."""
        if not timeline:
            return []

        # Sort the timeline by start time to process chronologically
        timeline.sort(key=lambda x: x["start_s"])

        merged = []
        if not timeline: return merged

        current_seg = timeline[0]

        for next_seg in timeline[1:]:
            is_same_speaker = (current_seg["speaker"] == next_seg["speaker"])
            # Check if the gap between segments is smaller than our configured max pause
            is_continuous = (next_seg["start_s"] - current_seg["end_s"] <= self.merge_max_pause_s)

            if is_same_speaker and is_continuous:
                # If same speaker and close in time, merge by extending the end time
                current_seg["end_s"] = max(current_seg["end_s"], next_seg["end_s"])
            else:
                # Otherwise, this segment is finished. Add it to the list and start a new one.
                merged.append(current_seg)
                current_seg = next_seg

        # Append the very last segment
        merged.append(current_seg)
        return merged
