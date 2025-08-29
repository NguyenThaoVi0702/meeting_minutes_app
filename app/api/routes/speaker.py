import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter, Depends, HTTPException, status, Query, Form, File, UploadFile
)
from pydantic import ValidationError

from app.processing.enrollment import SpeakerEnrollment
from app.schemas.speaker import (
    AllSpeakersResponse, SpeakerProfileInfo, SpeakerSearchResponse,
    SpeakerSearchResult, GenericSuccessResponse, SpeakerMetadataUpdate
)
from app.worker.tasks import get_enrollment_service # Re-using the singleton from the worker

logger = logging.getLogger(__name__)
router = APIRouter()

# ===================================================================
#   Dependency for Service Injection
# ===================================================================

def get_enrollment_manager() -> SpeakerEnrollment:
    """
    FastAPI dependency to get the singleton SpeakerEnrollment service.

    This ensures the heavy model is loaded only once and is shared across
    all API requests handled by this server process.
    """
    try:
        # The get_enrollment_service function from tasks.py handles the
        # singleton logic (initializing only if the global var is None).
        service = get_enrollment_service()
        if not service:
             raise RuntimeError("Enrollment service failed to initialize.")
        return service
    except Exception as e:
        logger.critical(f"FATAL: Could not provide SpeakerEnrollment service to endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speaker enrollment service is currently unavailable. Please check server logs."
        )

# ===================================================================
#   Speaker Collection Endpoints (List & Search)
# ===================================================================

@router.get(
    "/",
    response_model=AllSpeakersResponse,
    summary="List all enrolled speaker profiles"
)
async def list_all_speakers(
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Retrieves a list of all speaker profiles currently stored in the system,
    providing key metadata for each.
    """
    try:
        profiles_data = em.get_all_speaker_profiles() # This method needs to be implemented in enrollment.py
        
        # We use a Pydantic model to ensure the data shape is correct
        formatted_profiles = [SpeakerProfileInfo(**p) for p in profiles_data]

        return AllSpeakersResponse(data=formatted_profiles)
    except Exception as e:
        logger.error(f"Failed to list all speakers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching speaker list.")

@router.get(
    "/search",
    response_model=SpeakerSearchResponse,
    summary="Search for speaker profiles"
)
async def search_speaker_profiles(
    query: str = Query(..., min_length=1, description="The search term to find speakers by name or user_ad."),
    limit: int = Query(10, ge=1, le=50, description="The maximum number of results to return."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Searches for speaker profiles based on a query string. The search is performed
    against pre-indexed terms for display name and user_ad for fast results.
    """
    try:
        search_results_data = em.search_profiles(query.strip(), limit=limit)
        
        # Map the raw payload data to our clean API schema
        formatted_results = [SpeakerSearchResult(**r) for r in search_results_data]

        return SpeakerSearchResponse(data=formatted_results)
    except Exception as e:
        logger.error(f"Failed during speaker search for query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during search.")


# ===================================================================
#   Speaker Creation Endpoint
# ===================================================================

@router.post(
    "/",
    response_model=GenericSuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a new speaker profile"
)
async def enroll_new_speaker(
    metadata_json: str = Form(..., alias="metadata", description="A JSON string with speaker metadata (display_name, user_ad)."),
    files: List[UploadFile] = File(..., description="One or more audio files (.wav, .mp3) for the speaker's voice sample."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Creates a new speaker profile from audio samples and metadata.

    This endpoint handles multipart/form-data, processes the audio files to
    create a voice embedding, and stores it in the vector database.
    """
    try:
        metadata = SpeakerMetadataUpdate.model_validate_json(metadata_json)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid metadata JSON: {e}")

    user_ad = metadata.user_ad.lower().strip()
    if not user_ad:
         raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="user_ad is a required field in metadata.")

    # Create a temporary directory to store uploaded files before processing
    temp_dir = Path(settings.SHARED_AUDIO_PATH) / "temp_enrollment" / str(uuid.uuid4())
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_paths = []
    try:
        # Save all uploaded files to the temporary directory
        for file in files:
            if not file.filename: continue
            file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(file_path))

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid audio files were provided for enrollment.")

        # Call the enrollment service with the paths to the saved files
        em.enroll_new_speaker(
            user_ad=user_ad,
            audio_sample_paths=saved_paths,
            metadata=metadata.model_dump()
        )

        return GenericSuccessResponse(message=f"Speaker '{user_ad}' enrolled successfully.")

    except ValueError as e:
        # Catches specific errors from the service, like "speaker already exists".
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Enrollment failed for user_ad '{user_ad}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during enrollment.")
    finally:
        # CRITICAL: Always clean up the temporary files, even if an error occurred.
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# --- End of Part 1 ---


# ... (Continuing from Part 1) ...

# ===================================================================
#   Individual Speaker Resource Endpoints (Get, Update, Delete)
# ===================================================================

@router.get(
    "/{user_ad}",
    response_model=SpeakerProfileResponse,
    summary="Get a specific speaker's profile"
)
async def get_speaker_profile_details(
    user_ad: str = Path(..., description="The unique user_ad of the speaker to retrieve."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Retrieves the detailed profile for a single speaker by their reference ID (user_ad),
    including the internal Qdrant point details.
    """
    profile_record = em.get_profile_by_ref_id(user_ad)
    if not profile_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker with user_ad '{user_ad}' not found."
        )

    # We use a dedicated schema to structure the detailed response
    profile_details = QdrantPointDetails(
        qdrant_point_id=profile_record.id,
        payload=profile_record.payload,
        has_vector=bool(profile_record.vector)
    )

    return SpeakerProfileResponse(
        user_ad=user_ad,
        profile_details=profile_details
    )

@router.put(
    "/{user_ad}/metadata",
    response_model=GenericSuccessResponse,
    summary="Update a speaker's metadata"
)
async def update_speaker_metadata(
    metadata: SpeakerMetadataUpdate,
    user_ad: str = Path(..., description="The unique user_ad of the speaker to update."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Updates the metadata (e.g., display_name) for an existing speaker profile.
    This will also regenerate the search terms for the speaker.
    """
    try:
        # The service method will raise ValueError if the speaker doesn't exist.
        em.update_metadata(user_ad, metadata.model_dump(exclude_unset=True))
        return GenericSuccessResponse(message=f"Metadata for '{user_ad}' updated successfully.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update metadata for speaker '{user_ad}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while updating metadata.")


@router.post(
    "/{user_ad}/samples",
    response_model=GenericSuccessResponse,
    summary="Add new voice samples to a profile"
)
async def add_voice_samples_to_profile(
    user_ad: str = Path(..., description="The unique user_ad of the speaker to add samples to."),
    files: List[UploadFile] = File(..., description="One or more new audio files to add to the speaker's voice profile."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Adds new voice samples to an existing speaker's profile.

    The new samples' embeddings will be averaged with the existing embedding to
    refine and improve the speaker's voice print.
    """
    if not files or all(not f.filename for f in files):
        raise HTTPException(status_code=400, detail="No valid audio sample files provided.")

    temp_dir = Path(settings.SHARED_AUDIO_PATH) / "temp_samples" / str(uuid.uuid4())
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_paths = []
    try:
        for file in files:
            if not file.filename: continue
            file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(file_path))

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid audio files were processed.")
        
        # The service method will raise ValueError for not found or if samples are invalid
        em.add_samples_to_profile(user_ad, saved_paths)

        return GenericSuccessResponse(
            message=f"Successfully added {len(saved_paths)} new sample(s) to profile '{user_ad}'."
        )
    except ValueError as e:
        # Catches both "profile not found" and "could not extract embeddings" errors
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add samples for speaker '{user_ad}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred while adding samples.")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

@router.delete(
    "/{user_ad}",
    response_model=GenericSuccessResponse,
    summary="Delete a speaker's profile"
)
async def delete_speaker_profile(
    user_ad: str = Path(..., description="The unique user_ad of the speaker to delete."),
    em: SpeakerEnrollment = Depends(get_enrollment_manager)
):
    """
    Permanently deletes a speaker's profile, including their voice embedding
    and all associated metadata. This action cannot be undone.
    """
    try:
        success = em.remove_profile(user_ad)
        if not success:
            # This case handles if the profile was deleted between a check and this call.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Speaker with user_ad '{user_ad}' not found, nothing to delete."
            )
        return GenericSuccessResponse(message=f"Profile for '{user_ad}' was successfully deleted.")
    except Exception as e:
        logger.error(f"Failed to delete speaker '{user_ad}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while deleting the profile.")

# --- End of Part 2 ---