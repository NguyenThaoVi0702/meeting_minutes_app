import os
import time
import logging
import uuid
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import librosa
import soundfile as sf
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
from qdrant_client import QdrantClient, models

from app.core.config import settings

logger = logging.getLogger(__name__)

# ===================================================================
#   Helper Functions for Enhanced Search
# ===================================================================

def get_text_prefixes(text: str) -> List[str]:
    """Generates all possible prefixes for a given string. E.g., 'word' -> ['w', 'wo', 'wor', 'word']"""
    return [text[:i+1] for i in range(len(text))]

def generate_all_search_terms(display_name: str, user_ad: str) -> List[str]:
    """
    Generates a comprehensive list of search terms for a speaker, including prefixes
    for both accented and un-accented versions of their name and user_ad.
    This enables powerful and flexible 'search-as-you-type' functionality.
    """
    full_text = f"{display_name} {user_ad}"

    # Normalize text to remove accents (e.g., "Toàn Thắng" -> "Toan Thang")
    normalized_text = ''.join(
        c for c in unicodedata.normalize('NFD', full_text)
        if unicodedata.category(c) != 'Mn'
    )

    all_terms = set()
    for text_version in [full_text, normalized_text]:
        words = text_version.lower().split()
        for word in words:
            all_terms.update(get_text_prefixes(word))

    return list(all_terms)

# ===================================================================
#   Main Speaker Enrollment Service Class
# ===================================================================

class SpeakerEnrollment:
    """
    Manages speaker enrollment, voice embedding, and interaction with the
    Qdrant vector database. Designed to be instantiated as a long-lived
    service in a background worker.
    """
    QDRANT_VECTOR_SIZE = 512  # Based on the titanet-large model

    def __init__(self):
        """
        Initializes the service, loading the embedding model and connecting to Qdrant.
        """
        self.qdrant_collection_name = settings.QDRANT_COLLECTION_NAME
        self.target_sr = 16000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SpeakerEnrollment service is using device: {self.device}")

        self.embedding_model = self._load_embedding_model(settings.RIMECASTER_MODEL_PATH)

        logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
        self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=20)
        self._ensure_qdrant_collection()

    def _load_embedding_model(self, model_path: str):
        """Loads the NeMo speaker embedding model from the specified path."""
        logger.info(f"Loading NeMo Speaker Embedding model from: {model_path}")
        if not model_path or not os.path.exists(model_path):
             msg = f"NeMo model path '{model_path}' is invalid or does not exist."
             logger.critical(msg)
             raise FileNotFoundError(msg)
        try:
            model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path, map_location=self.device)
            model.eval()
            logger.info(f"Speaker Embedding model '{model_path}' loaded successfully.")
            return model
        except Exception as e:
            logger.critical(f"FATAL: Failed to load NeMo Speaker Embedding model: {e}", exc_info=True)
            raise

    def _ensure_qdrant_collection(self):
        """Ensures the Qdrant collection exists and has the correct configuration."""
        try:
            self.qdrant_client.get_collection(collection_name=self.qdrant_collection_name)
            logger.info(f"Qdrant collection '{self.qdrant_collection_name}' already exists.")
        except Exception:
            logger.warning(f"Qdrant collection '{self.qdrant_collection_name}' not found. Creating it now.")
            self.qdrant_client.recreate_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=models.VectorParams(size=self.QDRANT_VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            self.qdrant_client.create_payload_index(
                collection_name=self.qdrant_collection_name,
                field_name="search_terms",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True
            )
            logger.info(f"Qdrant collection '{self.qdrant_collection_name}' and keyword index created.")

    def get_embedding_from_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Extracts a speaker embedding directly from a NumPy audio waveform."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            sf.write(tmpfile.name, audio_data, self.target_sr)
            # The model's get_embedding method expects a file path
            embedding = self.embedding_model.get_embedding(tmpfile.name).squeeze().cpu().numpy()
        return embedding

    def _preprocess_and_embed_samples(self, audio_paths: List[str]) -> List[np.ndarray]:
        """Processes a list of audio files and returns a list of their embeddings."""
        embeddings = []
        for path in audio_paths:
            try:
                # Resample to the model's target sample rate
                audio, _ = librosa.load(path, sr=self.target_sr, mono=True)
                embedding = self.get_embedding_from_audio(audio)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Could not process or get embedding for sample {path}: {e}")
        return embeddings

    def enroll_new_speaker(self, user_ad: str, audio_sample_paths: List[str], metadata: Dict) -> str:
        """Enrolls a new speaker by creating an average embedding and storing it in Qdrant."""
        logger.info(f"Starting enrollment for new speaker: '{user_ad}'")
        if self.get_profile_by_ref_id(user_ad):
            raise ValueError(f"Speaker with user_ad '{user_ad}' already exists.")

        embeddings = self._preprocess_and_embed_samples(audio_sample_paths)
        if not embeddings:
            raise ValueError(f"Enrollment failed for '{user_ad}': No valid embeddings could be extracted from samples.")

        average_embedding = np.mean(np.array(embeddings), axis=0).tolist()
        new_point_id = str(uuid.uuid4())
        
        payload = metadata.copy()
        payload["user_ad"] = user_ad
        payload["enrolled_at_utc"] = datetime.utcnow().isoformat()
        payload["num_enrollment_samples"] = len(embeddings)
        payload["search_terms"] = generate_all_search_terms(metadata.get("display_name", ""), user_ad)

        try:
            self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=[models.PointStruct(id=new_point_id, vector=average_embedding, payload=payload)],
                wait=True
            )
            logger.info(f"Enrollment successful for '{user_ad}'. Qdrant Point ID: {new_point_id}")
            return new_point_id
        except Exception as e:
            logger.error(f"Enrollment failed for '{user_ad}': Error during Qdrant upsert.", exc_info=True)
            raise RuntimeError("Could not save speaker profile to vector database.")

    def get_profile_by_ref_id(self, user_ad: str) -> Optional[models.Record]:
        """Retrieves a speaker's full profile (including vector) by their user_ad."""
        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection_name,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="user_ad", match=models.MatchValue(value=user_ad))
                ]),
                limit=1, with_vectors=True
            )
            return points[0] if points else None
        except Exception as e:
            logger.error(f"Error retrieving profile for '{user_ad}': {e}", exc_info=True)
            return None

    def remove_profile(self, user_ad: str) -> bool:
        """Removes a speaker's profile from Qdrant."""
        logger.info(f"Attempting to remove profile for speaker: '{user_ad}'")
        profile = self.get_profile_by_ref_id(user_ad)
        if not profile:
            logger.warning(f"Profile for '{user_ad}' not found. Nothing to remove.")
            return False

        self.qdrant_client.delete(
            collection_name=self.qdrant_collection_name,
            points_selector=models.PointIdsList(points=[profile.id]),
            wait=True
        )
        logger.info(f"Successfully removed profile for '{user_ad}' (Point ID: {profile.id}).")
        return True

    def update_metadata(self, user_ad: str, new_metadata: Dict) -> bool:
        """Updates the metadata payload for an existing speaker."""
        logger.info(f"Attempting to update metadata for speaker: '{user_ad}'")
        profile = self.get_profile_by_ref_id(user_ad)
        if not profile:
            raise ValueError(f"Profile for '{user_ad}' not found, cannot update metadata.")

        updated_payload = profile.payload.copy()
        updated_payload.update(new_metadata)
        updated_payload["search_terms"] = generate_all_search_terms(
            updated_payload.get("display_name", ""), user_ad
        )

        self.qdrant_client.set_payload(
            collection_name=self.qdrant_collection_name,
            payload=updated_payload,
            points=[profile.id],
            wait=True
        )
        logger.info(f"Successfully updated metadata for '{user_ad}'.")
        return True

    def add_samples_to_profile(self, user_ad: str, new_audio_paths: List[str]) -> bool:
        """Adds new voice samples to an existing profile, recalculating the average embedding."""
        logger.info(f"Attempting to add new samples to speaker '{user_ad}'")
        profile = self.get_profile_by_ref_id(user_ad)
        if not profile:
            raise ValueError(f"Profile for '{user_ad}' not found, cannot add samples.")

        new_embeddings = self._preprocess_and_embed_samples(new_audio_paths)
        if not new_embeddings:
            raise ValueError("Could not extract any valid embeddings from the new audio samples.")

        old_vector = np.array(profile.vector)
        old_count = profile.payload.get("num_enrollment_samples", 1)
        new_count = len(new_embeddings)

        # Recalculate the weighted average of the embeddings
        new_vector_sum = np.sum(np.array(new_embeddings), axis=0)
        updated_vector = ((old_vector * old_count) + new_vector_sum) / (old_count + new_count)

        updated_payload = profile.payload
        updated_payload["num_enrollment_samples"] = old_count + new_count

        self.qdrant_client.upsert(
            collection_name=self.qdrant_collection_name,
            points=[models.PointStruct(id=profile.id, vector=updated_vector.tolist(), payload=updated_payload)],
            wait=True
        )
        logger.info(f"Successfully added {new_count} samples to '{user_ad}'. New total samples: {old_count + new_count}.")
        return True

    def get_all_enrolled_profiles_for_diarization(self) -> List[Dict]:
        """Retrieves all speaker profiles with their vectors, formatted for the diarization process."""
        try:
            all_points, _ = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection_name,
                limit=10000, with_payload=True, with_vectors=True
            )
            profiles = [{
                "embedding": np.array(point.vector),
                "payload": point.payload
            } for point in all_points if point.payload and "user_ad" in point.payload and point.vector]
            logger.info(f"Fetched {len(profiles)} valid profiles from Qdrant for diarization.")
            return profiles
        except Exception as e:
            logger.error(f"Error fetching all profiles from Qdrant: {e}", exc_info=True)
            return []

    def search_profiles(self, query: str, limit: int = 10) -> List[Dict]:
        """Searches for speaker profiles by matching the query against indexed search terms."""
        logger.info(f"Searching for profiles matching query: '{query}'")
        lower_query = query.lower().strip()
        if not lower_query: return []

        try:
            # Match against the pre-generated search term prefixes
            search_filter = models.Filter(must=[
                models.FieldCondition(key="search_terms", match=models.MatchValue(value=lower_query))
            ])
            hits = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_filter=search_filter,
                query_vector=[0.0] * self.QDRANT_VECTOR_SIZE, # Dummy vector for filtered search
                limit=limit,
                with_vectors=False
            )
            return [hit.payload for hit in hits]
        except Exception as e:
            logger.error(f"Error searching profiles with query '{query}': {e}", exc_info=True)
            return []