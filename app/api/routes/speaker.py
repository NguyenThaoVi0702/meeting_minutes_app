import os
import uuid
import shutil
import logging
from pathlib import Path as FilePath
from typing import List

from fastapi import (
    APIRouter, HTTPException, status, Query, Form, File, UploadFile, Path
)
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import UpdateStatus

from app.core.config import settings
from app.schemas.speaker import (
    AllSpeakersResponse, SpeakerProfileInfo, SpeakerSearchResponse,
    SpeakerSearchResult, GenericSuccessResponse, SpeakerMetadataUpdate,
    QdrantPointDetails, SpeakerProfileResponse
)
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter()


qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=10)

# ===================================================================
#   Speaker Collection Endpoints (List & Search)
# ===================================================================

@router.get(
    "/",
    response_model=AllSpeakersResponse,
    summary="List all enrolled speaker profiles"
)
async def list_all_speakers():
    """
    Retrieves a list of all speaker profiles currently stored in the system.
    This is a lightweight operation that only queries the vector database metadata.
    """
    try:
        all_points, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False 
        )
        profiles_data = [point.payload for point in all_points if point.payload]
        formatted_profiles = [SpeakerProfileInfo(**p) for p in profiles_data]
        return AllSpeakersResponse(data=formatted_profiles)
    except Exception as e:
        logger.error(f"Failed to list all speakers from Qdrant: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Could not connect to the vector database.")

@router.get(
    "/search",
    response_model=SpeakerSearchResponse,
    summary="Search for speaker profiles"
)
async def search_speaker_profiles(
    query: str = Query(..., min_length=1, description="The search term to find speakers by name or user_ad."),
    limit: int = Query(10, ge=1, le=50, description="The maximum number of results to return.")
):
    """
    Searches for speaker profiles based on a query string using the vector DB's indexed fields.
    """
    try:
        search_results = qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="search_terms", match=models.MatchValue(value=query.lower().strip()))
            ]),
            query_vector=[0.0] * 512, # Dummy vector required for filtered search
            limit=limit,
            with_vectors=False
        )
        formatted_results = [SpeakerSearchResult(**hit.payload) for hit in search_results]
        return SpeakerSearchResponse(data=formatted_results)
    except Exception as e:
        logger.error(f"Failed during speaker search for query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="An error occurred during search.")

# ===================================================================
#   Speaker Creation Endpoint
# ===================================================================

@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenericSuccessResponse,
    summary="Enroll a new speaker profile"
)
async def enroll_new_speaker(
    metadata_json: str = Form(..., alias="metadata"),
    files: List[UploadFile] = File(...)
):
    """
    Accepts a new speaker enrollment request, saves the audio samples,
    and queues the heavy processing (embedding generation) to a background worker.
    """
    try:
        metadata = SpeakerMetadataUpdate.model_validate_json(metadata_json)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid metadata JSON: {e}")

    user_ad = metadata.user_ad.lower().strip()
    if not user_ad:
         raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="user_ad is a required field in metadata.")

    # Lightweight check to prevent duplicate requests
    try:
        points, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="user_ad", match=models.MatchValue(value=user_ad))]),
            limit=1
        )
        if points:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Speaker with user_ad '{user_ad}' already exists.")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not verify speaker existence: {e}")

    # Save files to a permanent location for the worker to access
    enrollment_dir = FilePath(settings.ENROLLMENT_SAMPLES_PATH) / user_ad
    os.makedirs(enrollment_dir, exist_ok=True)
    
    saved_paths = []
    for file in files:
        if not file.filename: continue
        file_path = enrollment_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(str(file_path))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid audio files were provided for enrollment.")

    celery_app.send_task("enroll_speaker_task", args=[user_ad, saved_paths, metadata.model_dump()])

    return GenericSuccessResponse(message=f"Enrollment for speaker '{user_ad}' has been accepted for processing.")

# ===================================================================
#   Individual Speaker Resource Endpoints (Get, Update, Delete)
# ===================================================================

@router.get(
    "/{user_ad}",
    response_model=SpeakerProfileResponse,
    summary="Get a specific speaker's profile"
)
async def get_speaker_profile_details(
    user_ad: str = Path(..., description="The unique user_ad of the speaker to retrieve.")
):
    try:
        points, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="user_ad", match=models.MatchValue(value=user_ad))]),
            limit=1, with_vectors=True
        )
        if not points:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Speaker with user_ad '{user_ad}' not found.")
        
        profile_record = points[0]
        profile_details = QdrantPointDetails(
            qdrant_point_id=profile_record.id,
            payload=profile_record.payload,
            has_vector=bool(profile_record.vector)
        )
        return SpeakerProfileResponse(user_ad=user_ad, profile_details=profile_details)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not retrieve speaker profile: {e}")


@router.put(
    "/{user_ad}/metadata",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenericSuccessResponse,
    summary="Update a speaker's metadata (asynchronously)"
)
async def update_speaker_metadata(
    metadata: SpeakerMetadataUpdate,
    user_ad: str = Path(...)
):
    """
    Accepts a request to update a speaker's metadata and queues it for background processing.
    """
    celery_app.send_task("update_metadata_task", args=[user_ad, metadata.model_dump(exclude_unset=True)])
    return GenericSuccessResponse(message="Metadata update request has been accepted for processing.")


@router.post(
    "/{user_ad}/samples",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenericSuccessResponse,
    summary="Add new voice samples to a profile (asynchronously)"
)
async def add_voice_samples_to_profile(
    user_ad: str = Path(...),
    files: List[UploadFile] = File(...)
):
    """
    Accepts new voice samples for a speaker and queues the embedding and
    profile update to a background worker.
    """
    if not files or all(not f.filename for f in files):
        raise HTTPException(status_code=400, detail="No valid audio sample files provided.")
    samples_dir = FilePath(settings.ENROLLMENT_SAMPLES_PATH) / user_ad
    os.makedirs(samples_dir, exist_ok=True)
    
    saved_paths = []
    for file in files:
        if not file.filename: continue
        file_path = samples_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(str(file_path))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid audio files were processed.")

    celery_app.send_task("add_samples_task", args=[user_ad, saved_paths])
    return GenericSuccessResponse(message=f"Request to add {len(saved_paths)} new sample(s) has been accepted.")


@router.delete(
    "/{user_ad}",
    response_model=GenericSuccessResponse,
    summary="Delete a speaker's profile"
)
async def delete_speaker_profile(
    user_ad: str = Path(...)
):
    """
    Permanently deletes a speaker's profile from the vector database.
    """
    try:
        points, _ = qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="user_ad", match=models.MatchValue(value=user_ad))]),
            limit=1
        )
        if not points:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Speaker with user_ad '{user_ad}' not found.")
        
        point_id = points[0].id
        result = qdrant_client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[point_id]),
            wait=True
        )
        if result.status != UpdateStatus.COMPLETED:
             raise HTTPException(status_code=500, detail="Failed to delete profile from vector database.")
        
        return GenericSuccessResponse(message=f"Profile for '{user_ad}' was successfully deleted.")
    except HTTPException as e:
        raise e 
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"An error occurred while deleting the profile: {e}")
