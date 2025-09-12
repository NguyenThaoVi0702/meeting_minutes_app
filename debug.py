Of course. This is an excellent requirement that adds crucial, real-world metadata to your meeting records. The implementation is straightforward and touches three key areas:

Database Model: To store the full timestamps.

Upload Endpoint: To capture the timestamps at the exact moments the first and last chunks are received.

Document Generation Endpoint: To format the stored timestamps and provide them as context to the LLM.

Here is the step-by-step guide and the complete code for each part.

Step 1: Update the Database Model

First, we need to add fields to the MeetingJob model to store the full, precise timestamps.

Open app/db/models.py and add upload_started_at and upload_finished_at.

code
Python
download
content_copy
expand_less

# in app/db/models.py

class MeetingJob(SQLModel, table=True):
    # ... (id, request_id, user_id, etc.) ...
    language: str = Field(default="vi", description="The currently active language for this meeting.")

    # --- NEW TIMESTAMP FIELDS ---
    upload_started_at: Optional[datetime] = Field(default=None, description="Timestamp of when the first audio chunk was received.")
    upload_finished_at: Optional[datetime] = Field(default=None, description="Timestamp of when the last audio chunk was received.")
    # --- END OF NEW FIELDS ---

    # --- Job State & Workflow ---
    status: str = Field(default="uploading", index=True,
                        description="Tracks the current state of the job...")
    # ... (rest of the model) ...

After adding this, when you restart your application with a new database, this schema will be applied. If you are modifying an existing database, you would need a migration tool like Alembic.

Step 2: Capture Timestamps in the Upload Endpoint

Next, we modify the upload_file_chunk endpoint in app/api/routes/meeting.py to record these timestamps.

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# in app/api/routes/meeting.py

from datetime import datetime # Make sure this is imported at the top of the file

# ... (other code) ...

@router.post("/upload-file-chunk", status_code=status.HTTP_202_ACCEPTED, summary="Upload a single audio chunk")
async def upload_file_chunk(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db_session),
    requestId: str = Form(...),
    isLastChunk: bool = Form(...),
    FileData: UploadFile = File(...),
):
    job = session.exec(select(MeetingJob).where(MeetingJob.request_id == requestId)).first()
    if not job:
        raise HTTPException(status_code=404, detail="Meeting job not found.")
    if job.status != "uploading":
        raise HTTPException(status_code=400, detail=f"Cannot upload chunks when job status is '{job.status}'.")

    # --- CAPTURE START TIME ---
    # If the start time hasn't been set yet, this must be the first chunk.
    if not job.upload_started_at:
        job.upload_started_at = datetime.utcnow()
        logger.info(f"First chunk received for '{requestId}'. Recording start time.")
    # --- END CAPTURE START TIME ---

    session_dir = Path(settings.SHARED_AUDIO_PATH) / requestId
    chunk_path = session_dir / FileData.filename
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(FileData.file, buffer)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chunk file: {e}")

    if isLastChunk:
        job.status = "assembling"
        
        # --- CAPTURE END TIME ---
        job.upload_finished_at = datetime.utcnow()
        logger.info(f"Last chunk received for '{requestId}'. Recording end time.")
        # --- END CAPTURE END TIME ---

        session.add(job)
        session.commit() # Commit the timestamps to the database
        
        logger.info(f"Last chunk received for '{requestId}'. Triggering background assembly and transcription.")
        await websocket_manager.broadcast_to_job(requestId, {"status": "assembling"})
        background_tasks.add_task(assemble_and_transcribe, requestId, job.language)
    else:
        # Commit the start time on the first chunk without waiting for the last
        session.add(job)
        session.commit()

    return {"status": 202, "message": f"Chunk '{FileData.filename}' accepted."}
Step 3: Provide Formatted Timestamps to the LLM

Now, we update the generate_and_download_document endpoint in app/api/routes/meeting.py. Instead of asking the LLM to guess the time, we will provide it as explicit context, which is far more reliable and token-efficient.

The LLM prompt already asks for start_time and end_time in its JSON structure, so we just need to provide it the information to fill those fields accurately.

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# in app/api/routes/meeting.py (Part 2)

# ... (other code) ...

@router.get(
    "/{request_id}/download/document",
    summary="Generate and download a formal meeting document"
)
async def generate_and_download_document(
    job: MeetingJob = Depends(get_job_with_any_transcript),
    template_type: str = Query(..., enum=["bbh_hdqt", "nghi_quyet"], description="The type of document template to use."),
    db: Session = Depends(get_db_session)
):
    # 1. Get the source transcript (code is unchanged)
    transcription_entry = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()
    transcript_text = "\n".join([seg['text'] for seg in transcription_entry.transcript_data])

    # --- PREPARE CONTEXT FOR LLM ---
    # Format the stored datetimes into the required "HH:MM" string format.
    # Provide a default if for some reason the timestamps weren't recorded.
    start_time_str = job.upload_started_at.strftime('%H:%M') if job.upload_started_at else "N/A"
    end_time_str = job.upload_finished_at.strftime('%H:%M') if job.upload_finished_at else "N/A"
    
    # Prepend this context to the main transcript. The LLM will use this
    # information to fill the JSON fields accurately.
    context_header = (
        f"**THÔNG TIN BỐI CẢNH CUỘC HỌP:**\n"
        f"- Giờ bắt đầu: {start_time_str}\n"
        f"- Giờ kết thúc: {end_time_str}\n\n"
        f"**NỘI DUNG BIÊN BẢN (TRANSCRIPT):**\n"
    )
    full_llm_input = context_header + transcript_text
    # --- END PREPARE CONTEXT ---

    # 2. Call AI service to get structured JSON
    try:
        task_name = f"summary_{template_type}"
        # Pass the full input with the context header to the LLM
        llm_json_response = await ai_service.get_response(task=task_name, user_message=full_llm_input)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get data from AI service: {e}")

    # 3. Generate the document (code is unchanged)
    try:
        document_buffer = generate_templated_document(template_type, llm_json_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate the document: {e}")

    # 4. Return the file to the user (code is unchanged)
    filename = f"{template_type}_{job.bbh_name.replace(' ', '_')}.docx"
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"}
    
    return StreamingResponse(
        document_buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers
    )

With these three changes, your system now correctly:

Stores the precise start and end times of the audio upload.

Captures these times automatically during the upload process.

Formats and provides this information reliably to the LLM, ensuring the final generated DOCX contains the correct HH:MM times as requested.
