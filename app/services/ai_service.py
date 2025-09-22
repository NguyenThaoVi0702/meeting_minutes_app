Of course. Let's get the timezone fixed permanently. My apologies that the previous explanation wasn't clear enough. Here is a definitive, step-by-step guide to ensure the time displayed in your documents is always correct for the Ho Chi Minh timezone.

The Core Problem Explained Simply

Your database correctly stores time in UTC (Coordinated Universal Time), which is the universal standard for servers. This is a good thing.

Your Python code retrieves this UTC time, but it's "naive"—it doesn't have a timezone label attached.

When you format this naive time directly with strftime('%H:%M'), it simply prints the UTC hour and minute, which is 7 hours behind Ho Chi Minh time.

The solution is to tell Python, "This time I got from the database is UTC, now please convert it to the Asia/Ho_Chi_Minh timezone before you turn it into a string."

Step-by-Step Fix

You only need to modify one file: app/api/routes/meeting.py.

Step 1: Add the Necessary Imports

At the very top of app/api/routes/meeting.py, add two new imports. One is for handling timezone information, and the other is for the UTC timezone constant.

code
Python
download
content_copy
expand_less

# In /app/api/routes/meeting.py

import json
import logging
import os
import shutil
# <<< ADD THESE TWO LINES AT THE TOP WITH OTHER IMPORTS >>>
from datetime import timezone
from zoneinfo import ZoneInfo
# ---
from datetime import datetime
from pathlib import Path
# ... rest of the imports
Step 2: Replace the generate_and_download_document function

Replace your entire generate_and_download_document function with the version below. This new version contains the correct logic to handle the timezone conversion safely. The changes are clearly marked.

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# In /app/api/routes/meeting.py

@router.get("/{request_id}/download/document", summary="Generate and download a formal meeting document")
async def generate_and_download_document(
    job: MeetingJob = Depends(get_job_with_any_transcript),
    template_type: str = Query(..., enum=["bbh_hdqt", "nghi_quyet"], description="The type of document template to use."),
    db: Session = Depends(get_db_session)
):
    transcription_entry = db.exec(
        select(Transcription).where(
            Transcription.meeting_job_id == job.id,
            Transcription.language == job.language
        )
    ).first()
    transcript_text = "\n".join([seg['text'] for seg in transcription_entry.transcript_data])

    # <<< START OF TIMEZONE FIX >>>

    # 1. Define our target timezone
    local_tz = ZoneInfo("Asia/Ho_Chi_Minh")

    # 2. Get the naive UTC datetimes from the database
    start_time_utc_naive = job.upload_started_at
    end_time_utc_naive = job.upload_finished_at

    # 3. Make the datetimes "aware" that they are UTC, then convert to local time
    start_time_local = None
    if start_time_utc_naive:
        start_time_local = start_time_utc_naive.replace(tzinfo=timezone.utc).astimezone(local_tz)

    end_time_local = None
    if end_time_utc_naive:
        end_time_local = end_time_utc_naive.replace(tzinfo=timezone.utc).astimezone(local_tz)

    # 4. Format the CORRECT local time into strings for the AI prompt
    meeting_date_str = start_time_local.strftime('%d/%m/%Y') if start_time_local else "N/A"
    start_time_str = start_time_local.strftime('%H:%M') if start_time_local else "N/A"
    end_time_str = end_time_local.strftime('%H:%M') if end_time_local else "N/A"
    
    # <<< END OF TIMEZONE FIX >>>

    context_header = (
        f"**THÔNG TIN BỐI CẢNH CUỘC HỌP:**\n"
        f"- Ngày họp: {meeting_date_str}\n"
        f"- Giờ bắt đầu: {start_time_str}\n"
        f"- Giờ kết thúc: {end_time_str}\n\n"
        f"**NỘI DUNG BIÊN BẢN (TRANSCRIPT):**\n"
    )
    full_llm_input = context_header + transcript_text
    
    try:
        task_name = f"summary_{template_type}"
        llm_json_response = await ai_service.get_response(task=task_name, user_message=full_llm_input)
        logger.info(f"RAW AI JSON RESPONSE for '{template_type}':\n{llm_json_response}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to get data from AI service: {e}")

    try:
        document_buffer = generate_templated_document(template_type, llm_json_response)
    except Exception as e:
        logger.error(f"Error during document generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate the document: {e}")

    filename = f"{template_type}_{job.bbh_name.replace(' ', '_')}.docx"
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"}
    
    return StreamingResponse(
        document_buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers
    )
Summary of Actions to Take

Modify the Code: Apply the changes to app/api/routes/meeting.py as shown above.

Rebuild the Docker Image: Because you have changed a Python file, you must rebuild the image to include the changes.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
docker-compose down
docker-compose up --build -d

Run your test script again.

Now, when the document generation endpoint is called, it will correctly convert the UTC time from the database to Ho Chi Minh time before sending it to the LLM. The final documents will have the correct local time and date.
