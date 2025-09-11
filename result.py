Of course. This is an excellent use case for automating a full workflow. The plan will be:

Identify Meeting 1 Chunks: The script will first identify all audio chunks belonging to the 9:30 AM - 11:05 AM time slot, just as before.

Generate a Unique ID: It will create a unique session_id for this entire meeting process. This is crucial for the API to link all the chunks, the final transcript, and the summary together.

Transcribe Chunks: It will loop through each Meeting 1 chunk in chronological order and send it to the /transcriptions endpoint.

Combine Transcript: After all chunks are sent, it will call the /get-all-segment endpoint to get the full, combined text.

Summarize Text: It will send this combined text to the /summary endpoint.

Download Report: Finally, it will call the /download-word endpoint to get the .docx file and save it to the same folder as app.py.

You will need the requests library for this. We will add it to the Dockerfile.

Step 1: Update the Dockerfile

Add requests to the pip install line.

code
Dockerfile
download
content_copy
expand_less

# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container at /app
COPY app.py .

# Install pydub AND requests (using Nexus config if needed)
RUN pip install --no-cache-dir pydub requests # --extra-index-url https://your-nexus-repo/repository/pypi-all/simple/

# Command to run the application
CMD ["python", "app.py"]
Step 2: Replace app.py with the New Code

This new script contains all the logic for identifying files and then calling the API endpoints in the correct sequence.

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
import os
import uuid
import time
import requests
import io
from pydub import AudioSegment
from datetime import datetime, timedelta

# --- API Configuration ---
API_BASE_URL = "http://10.43.128.107:8086"
USER_ID = "automated_processor" # A static user ID for this script

def process_meeting_with_api(meeting_chunks, output_folder_path, meeting_name, target_date_str):
    """
    Takes a list of audio chunk filepaths for a single meeting, sends them to the API
    for transcription, summarization, and downloads the final DOCX report.
    """
    if not meeting_chunks:
        print(f"No audio chunks found for {meeting_name}. Skipping API processing.")
        return

    # Generate a single, unique session_id for this entire meeting
    session_id = str(uuid.uuid4())
    print(f"\n--- Starting API processing for {meeting_name} with Session ID: {session_id} ---")

    # 1. Transcribe each chunk sequentially
    print(f"\n[Step 1/4] Transcribing {len(meeting_chunks)} audio chunks...")
    for i, chunk_info in enumerate(meeting_chunks):
        filepath = chunk_info['filepath']
        filename = chunk_info['filename']
        print(f"  - Sending chunk {i+1}/{len(meeting_chunks)}: {filename}")
        
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'audio/wav')}
                data = {'session_id': session_id, 'user_id': USER_ID}
                response = requests.post(f"{API_BASE_URL}/transcriptions", files=files, data=data, timeout=300)
                
                if response.status_code == 200:
                    print(f"    -> Success. Transcript chunk received by server.")
                else:
                    print(f"    -> ERROR: Failed to send chunk. Status: {response.status_code}, Response: {response.text}")
                    # Decide if you want to stop or continue on error
                    continue 
        except requests.exceptions.RequestException as e:
            print(f"    -> FATAL ERROR: Network or connection error for chunk {filename}: {e}")
            return # Stop processing this meeting if a chunk fails critically
        time.sleep(1) # Small delay to avoid overwhelming the server

    # 2. Get the full combined transcript
    print("\n[Step 2/4] Retrieving full combined transcript...")
    full_text = ""
    try:
        data = {'session_id': session_id, 'user_id': USER_ID}
        # Note: API has @app.post but might be a GET. Assuming POST as per decorator.
        response = requests.post(f"{API_BASE_URL}/get-all-segment", data=data, timeout=120)
        
        if response.status_code == 200:
            full_text = response.json().get("full_text", "")
            if not full_text:
                print("  -> ERROR: API returned an empty transcript. Cannot proceed.")
                return
            print(f"  -> Success. Retrieved full transcript ({len(full_text)} characters).")
        else:
            print(f"  -> ERROR: Failed to get full transcript. Status: {response.status_code}, Response: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"  -> FATAL ERROR: Could not retrieve full transcript: {e}")
        return

    # 3. Summarize the full text
    print("\n[Step 3/4] Requesting summary of the transcript...")
    try:
        # The /summary endpoint expects a file. We'll send the text as an in-memory file.
        transcript_bytes = io.BytesIO(full_text.encode('utf-8'))
        files = {'file': ('transcript.txt', transcript_bytes, 'text/plain')}
        data = {'session_id': session_id, 'user_id': USER_ID}
        response = requests.post(f"{API_BASE_URL}/summary", files=files, data=data, timeout=300)

        if response.status_code == 200:
            print("  -> Success. Summary generation request accepted by server.")
        else:
            print(f"  -> ERROR: Failed to request summary. Status: {response.status_code}, Response: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"  -> FATAL ERROR: Could not request summary: {e}")
        return

    # 4. Download the final DOCX report
    print("\n[Step 4/4] Downloading the final DOCX report...")
    output_docx_path = os.path.join(output_folder_path, f"MeetingReport_TO_{target_date_str}_0930_1105.docx")
    try:
        data = {'session_id': session_id, 'user_id': USER_ID}
        response = requests.post(f"{API_BASE_URL}/download-word", data=data, timeout=120, stream=True)

        if response.status_code == 200:
            with open(output_docx_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  -> SUCCESS! Report saved to: {output_docx_path}")
        else:
            print(f"  -> ERROR: Failed to download DOCX file. Status: {response.status_code}, Response: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"  -> FATAL ERROR: Could not download DOCX report: {e}")
        return
        
    print(f"\n--- Finished processing for {meeting_name} ---")


def find_and_process_chunks(input_folder_path, output_folder_path, target_date_str="11092025"):
    """
    Finds audio chunks, sorts them into meetings by timestamp, and then processes
    Meeting 1 through the transcription and summary API.
    """
    chunk_data = []
    print("Scanning and collecting chunk data...")
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".wav") and f"TO {target_date_str}" in filename:
            filepath = os.path.join(input_folder_path, filename)
            try:
                mtime_timestamp = os.path.getmtime(filepath)
                mtime_datetime = datetime.fromtimestamp(mtime_timestamp)
                chunk_data.append({'filepath': filepath, 'filename': filename, 'mtime': mtime_datetime})
            except Exception as e:
                print(f"Could not process file {filename}: {e}")

    if not chunk_data:
        print(f"No audio files found for date {target_date_str}.")
        return

    chunk_data.sort(key=lambda x: x['mtime'])
    print(f"Found and sorted {len(chunk_data)} chunks by modification time.")

    # Define meeting time boundaries
    first_chunk_mtime = chunk_data[0]['mtime']
    recording_day_start = datetime(first_chunk_mtime.year, first_chunk_mtime.month, first_chunk_mtime.day)

    meeting1_start_td = timedelta(hours=9, minutes=30)
    meeting1_end_td   = timedelta(hours=11, minutes=5)
    
    meeting1_chunks = []

    for chunk in chunk_data:
        # We need pydub here just to check duration for accurate assignment
        try:
            audio = AudioSegment.from_wav(chunk['filepath'])
            duration_ms = len(audio)
            
            chunk_relative_start_td = chunk['mtime'] - recording_day_start
            chunk_relative_end_td = chunk_relative_start_td + timedelta(milliseconds=duration_ms)
            
            # Check for overlap with Meeting 1 (9h30-11h05)
            if chunk_relative_start_td < meeting1_end_td and chunk_relative_end_td > meeting1_start_td:
                meeting1_chunks.append(chunk)
        except Exception as e:
            print(f"Could not read duration for {chunk['filename']}, skipping assignment. Error: {e}")

    print(f"\nFound {len(meeting1_chunks)} chunks belonging to Meeting 1 (09:30 - 11:05).")

    # Now, process only Meeting 1 chunks with the API
    process_meeting_with_api(meeting1_chunks, output_folder_path, "Meeting 1", target_date_str)


if __name__ == "__main__":
    input_data_container_path = "/app/input_data"
    output_results_container_path = "/app"
    
    find_and_process_chunks(input_data_container_path, output_results_container_path, "11092025")
Step 3: Re-build and Run the Docker Container

The process to run this is exactly the same as before.

Re-build the Image: Navigate to your project folder (where app.py and Dockerfile are) and run the build command. This is important because it will install the requests library.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
docker build -t audio-processor-image .

Run the Container: Use the same docker run command with the two volume mounts.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
docker run --rm \
  -v /path/to/your/host_audio_data:/app/input_data \
  -v /path/to/my_audio_processor:/app \
  audio-processor-image

Remember to replace:

/path/to/your/host_audio_data with the absolute path to your folder containing the source .wav files.

/path/to/my_audio_processor with the absolute path to your project folder containing app.py.

What Will Happen Now

When you run the container, you will see a detailed log in your terminal:

It will first identify and sort all the audio chunks.

It will tell you how many chunks it found for Meeting 1.

It will then start the API workflow, printing its progress for each of the 4 steps.

If everything is successful, a file named MeetingReport_TO_11092025_0930_1105.docx will appear in your my_audio_processor folder on your host machine.

If any API call fails, it will print an error message and, in most cases, stop the process for that meeting.
