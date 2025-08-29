from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ===================================================================
#   General Purpose & Wrapper Models
# ===================================================================

class GenericSuccessResponse(BaseModel):
    """
    A standard response model for operations that don't need to return
    complex data, such as a successful deletion or update.
    """
    success: bool = True
    message: str

# ===================================================================
#   Speaker Search & Listing Models
# ===================================================================

class SpeakerSearchResult(BaseModel):
    """
    Represents a single speaker in a search result list. Provides essential,
    publicly viewable information.
    """
    display_name: str = Field(..., description="The speaker's display name.")
    user_ad: str = Field(..., description="The speaker's unique Active Directory username, used as the reference ID.")

class SpeakerSearchResponse(BaseModel):
    """
    The response model for a speaker search query.
    """
    success: bool = True
    message: str = "Search successful"
    data: List[SpeakerSearchResult]

class SpeakerProfileInfo(BaseModel):
    """
    Provides detailed metadata about a speaker's profile. Used when listing all speakers.
    """
    display_name: str
    user_ad: str
    enrolled_at_utc: Optional[datetime] = None
    num_enrollment_samples: Optional[int] = None

class AllSpeakersResponse(BaseModel):
    """
    The response model for the endpoint that lists all enrolled speakers.
    """
    success: bool = True
    message: str = "Successfully retrieved all speaker profiles."
    data: List[SpeakerProfileInfo]

# ===================================================================
#   Detailed Speaker Profile Models
# ===================================================================

class QdrantPointDetails(BaseModel):
    """
    A Pydantic model representing the detailed information stored for a
    speaker's vector profile in the Qdrant database.
    """
    qdrant_point_id: str = Field(..., description="The UUID of the point in Qdrant.")
    payload: Dict[str, Any] = Field(..., description="The metadata payload associated with the vector.")
    has_vector: bool = Field(..., description="Indicates if a voice embedding vector is present for this profile.")

class SpeakerProfileResponse(BaseModel):
    """
    The response model for fetching a single, detailed speaker profile.
    """
    success: bool = True
    message: str = "Profile retrieved successfully"
    user_ad: str = Field(..., description="The unique reference ID for the speaker.")
    profile_details: QdrantPointDetails

# ===================================================================
#   Input Models for Creating/Updating Speakers
# ===================================================================

class SpeakerMetadataUpdate(BaseModel):
    """

    Schema for the metadata provided when creating or updating a speaker profile.
    This is typically sent as a JSON string in a multipart/form-data request.
    """
    display_name: str = Field(..., description="The speaker's display name.", min_length=1)
    # The user_ad is typically taken from the URL path parameter, not the body,
    # but it can be included for validation or creation purposes.
    user_ad: Optional[str] = Field(None, description="The speaker's unique Active Directory username.")

    class Config:
        # Example for FastAPI's auto-generated documentation
        json_schema_extra = {
            "example": {
                "display_name": "Nguyen Van A",
                "user_ad": "anv12",
            }
        }