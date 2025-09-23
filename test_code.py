

import requests
import os
import time
import csv
import glob
import json
import threading
import websocket
from requests.exceptions import JSONDecodeError, RequestException
from urllib.parse import quote

# --- CONFIGURATION ---
API_BASE_URL = "http://api:8072/api/v1"
SPEAKER_METADATA_PATH = "speaker_metadata.csv"
SPEAKER_AUDIO_DIR = "speaker_audio"
MEETING_AUDIO_DIR = "meeting_to"
TEST_USERNAME = "test_user_01"
MAX_SAMPLES_PER_SPEAKER = 20
UPLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_DIR = "/test_client/downloads"


WEBSOCKET_DONE = threading.Event()

# --- HELPER FUNCTIONS (No changes needed here) ---
def print_response(response, step_name):
    print(f"  -> [{step_name}] Status: {response.status_code}")
    try:
        print(f"     Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except JSONDecodeError:
        print(f"     Response (non-JSON): {response.text[:500]}...")

def read_speaker_metadata():
    metadata = {}
    with open(SPEAKER_METADATA_PATH, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if 'folder_name' in row: metadata[row['folder_name']] = row
    print(f"Loaded metadata for {len(metadata)} speakers from CSV.")
    return metadata

def group_meeting_chunks():
    meetings = {}
    search_path = os.path.join(MEETING_AUDIO_DIR, '*.wav')
    for chunk_path in glob.glob(search_path):
        filename = os.path.basename(chunk_path)
        meeting_name = "_".join(filename.split('_')[:-1])
        if meeting_name not in meetings: meetings[meeting_name] = []
        meetings[meeting_name].append(chunk_path)
    for name, chunks in meetings.items():
        chunks.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    print(f"Found {len(meetings)} meetings to process.")
    return meetings

def find_audio_files(directory):
    patterns = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return files

def listen_on_websocket(request_id):
    encoded_request_id = quote(request_id)
    ws_url = f"{API_BASE_URL.replace('http', 'ws')}/meeting/ws/{encoded_request_id}"
    print(f"\n[WebSocket] Connecting to {ws_url}...")
    def on_message(ws, message):
        data = json.loads(message)
        status = data.get('status')
        print(f"\n[WebSocket] === STATUS UPDATE: {status} ===")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if status in ['transcription_complete', 'completed', 'failed', 'cancelled']:
            WEBSOCKET_DONE.set()
            ws.close()
    def on_error(ws, error): print(f"[WebSocket] Error: {error}"); WEBSOCKET_DONE.set()
    def on_close(ws, c, m): print("[WebSocket] Connection closed.")
    def on_open(ws): print("[WebSocket] Connection opened.")
    ws_app = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws_app.run_forever()

def run_full_test():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    metadata = read_speaker_metadata()
    meetings = group_meeting_chunks()
    enrolled_speakers = {}
    speaker_to_delete = None

    # ==================================================================
    # STEP 1: ENROLL SPEAKERS
    # ==================================================================
    print("\n--- STEP 1: ENROLLING SPEAKERS ---")
    speaker_folders = [d for d in os.listdir(SPEAKER_AUDIO_DIR) if os.path.isdir(os.path.join(SPEAKER_AUDIO_DIR, d))]
    for folder_name in speaker_folders:
        meta = metadata.get(folder_name)
        if not meta or not meta.get('user_ad'): continue
        user_ad = meta['user_ad']
        audio_files = find_audio_files(os.path.join(SPEAKER_AUDIO_DIR, folder_name))
        if not audio_files: continue
        files_to_upload_paths = audio_files[:MAX_SAMPLES_PER_SPEAKER]
        files_to_upload_tuples = [('files', (os.path.basename(f), open(f, 'rb'))) for f in files_to_upload_paths]
        payload = {'metadata': json.dumps({"display_name": meta.get('display_name', "Default Name"), "user_ad": user_ad})}
        try:
            response = requests.post(f"{API_BASE_URL}/speaker/", data=payload, files=files_to_upload_tuples, timeout=UPLOAD_TIMEOUT_SECONDS)
            if response.status_code == 202:
                enrolled_speakers[user_ad] = meta
                if speaker_to_delete is None: speaker_to_delete = user_ad
        except RequestException as e: print(f"  -> ERROR: Connection error: {e}")
        time.sleep(2)

    # ==================================================================
    # STEP 2: TEST SPEAKER MANAGEMENT (INCLUDING SEARCH)
    # ==================================================================
    print("\n\n--- STEP 2: TESTING SPEAKER MANAGEMENT & SEARCH ---")
    if enrolled_speakers:
        # Test Search Endpoint
        first_speaker_ad = list(enrolled_speakers.keys())[0]
        first_speaker_name_part = enrolled_speakers[first_speaker_ad].get('display_name', 'Default Name').split(' ')[0]
        print(f"\nTesting: GET /speaker/search?query={first_speaker_name_part}")
        search_response = requests.get(f"{API_BASE_URL}/speaker/search", params={'query': first_speaker_name_part})
        print_response(search_response, "Search Speaker")

        # Test Deletion
        if speaker_to_delete:
            print(f"\nTesting: DELETE /speaker/{speaker_to_delete}")
            delete_response = requests.delete(f"{API_BASE_URL}/speaker/{speaker_to_delete}")
            print_response(delete_response, "Delete Speaker")

    # ==================================================================
    # STEP 3 & 4: PROCESS A FULL MEETING
    # ==================================================================
    print("\n\n--- STEP 3 & 4: PROCESSING A FULL MEETING ---")
    if not meetings:
        print("No meetings found. Skipping meeting tests.")
        return
        
    meeting_name, chunks = list(meetings.items())[0]
    request_id = f"test_{meeting_name}_{int(time.time())}"
    encoded_request_id = quote(request_id)

    # Start meeting
    start_payload = { 'requestId': request_id, 'username': TEST_USERNAME, 'language': 'vi', 'filename': f"{meeting_name}.wav", 'bbhName': "Initial BBH Name", 'Type': "Initial Type", 'Host': "Initial Host" }
    requests.post(f"{API_BASE_URL}/meeting/start-bbh", data=start_payload)

    # Test Update Meeting Info Endpoint
    print(f"\nTesting: PATCH /meeting/{encoded_request_id}/info")
    update_info_payload = {"bbh_name": "Updated Quarterly Review", "meeting_host": "Updated CEO"}
    update_info_response = requests.patch(f"{API_BASE_URL}/meeting/{encoded_request_id}/info?username={TEST_USERNAME}", json=update_info_payload)
    print_response(update_info_response, "Update Meeting Info")

    # Upload chunks
    for i, chunk_path in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        chunk_payload = {'requestId': request_id, 'isLastChunk': str(is_last)}
        files = {'FileData': (os.path.basename(chunk_path), open(chunk_path, 'rb'), 'audio/wav')}
        requests.post(f"{API_BASE_URL}/meeting/upload-file-chunk", data=chunk_payload, files=files)

    # Wait for transcription
    print("\n--- WAITING FOR TRANSCRIPTION VIA WEBSOCKET ---")
    ws_thread = threading.Thread(target=listen_on_websocket, args=(request_id,))
    ws_thread.start()
    WEBSOCKET_DONE.wait(timeout=600)
    ws_thread.join()

    # Trigger and wait for diarization
    requests.post(f"{API_BASE_URL}/meeting/{encoded_request_id}/diarize?username={TEST_USERNAME}")
    WEBSOCKET_DONE.clear()
    print("\n--- WAITING FOR DIARIZATION VIA WEBSOCKET ---")
    ws_thread_2 = threading.Thread(target=listen_on_websocket, args=(request_id,))
    ws_thread_2.start()
    WEBSOCKET_DONE.wait(timeout=600)
    ws_thread_2.join()
    
    # ==================================================================
    # STEP 5: ANALYSIS AND DOWNLOADS (Primary Workflow)
    # ==================================================================
    print("\n\n--- STEP 5: TESTING ANALYSIS AND DOWNLOADS ---")
    audio_dl_response = requests.get(f"{API_BASE_URL}/meeting/{encoded_request_id}/download/audio?username={TEST_USERNAME}")
    if audio_dl_response.status_code == 200:
        audio_save_path = os.path.join(DOWNLOAD_DIR, f"{request_id}_audio.wav")
        with open(audio_save_path, 'wb') as f: f.write(audio_dl_response.content)
        print(f"  -> [Download Audio] OK. File saved inside container at {audio_save_path}")

    document_types_to_test = ["bbh_hdqt", "nghi_quyet"]
    for doc_type in document_types_to_test:
        print(f"\nTesting document download (type: {doc_type})")
        doc_params = {'username': TEST_USERNAME, 'template_type': doc_type}
        doc_dl_response = requests.get(f"{API_BASE_URL}/meeting/{encoded_request_id}/download/document", params=doc_params)
        if doc_dl_response.status_code == 200:
            doc_save_path = os.path.join(DOWNLOAD_DIR, f"{request_id}_{doc_type}.docx")
            with open(doc_save_path, 'wb') as f: f.write(doc_dl_response.content)
            print(f"  -> [Download Document] OK. File saved inside container at {doc_save_path}")
        else:
            print_response(doc_dl_response, f"Download {doc_type}")
    # ==================================================================
    # STEP 6: TEST EDITING, LANGUAGE CHANGE, AND SIDE EFFECTS
    # ==================================================================
    print("\n\n--- STEP 6: TESTING EDITING AND LANGUAGE CHANGE ---")

    # Test Change Language Endpoint
    print(f"\nTesting: POST /meeting/{encoded_request_id}/language (to 'en')")
    lang_change_payload = {"language": "en"}
    lang_change_response = requests.post(f"{API_BASE_URL}/meeting/{encoded_request_id}/language?username={TEST_USERNAME}", json=lang_change_payload)
    print_response(lang_change_response, "Change Language")

    # Wait for the new (English) transcription
    WEBSOCKET_DONE.clear()
    print("\n--- WAITING FOR NEW 'en' TRANSCRIPTION VIA WEBSOCKET ---")
    ws_thread_3 = threading.Thread(target=listen_on_websocket, args=(request_id,))
    ws_thread_3.start()
    WEBSOCKET_DONE.wait(timeout=600)
    ws_thread_3.join()

    # Test Update Plain Transcript Endpoint
    print(f"\nTesting: PUT /meeting/{encoded_request_id}/transcript/plain")
    status_res = requests.get(f"{API_BASE_URL}/meeting/{encoded_request_id}/status?username={TEST_USERNAME}")
    if status_res.status_code == 200 and status_res.json()['data']['plain_transcript']:
        original_transcript = status_res.json()['data']['plain_transcript']
        # Create a modified version
        modified_segments = original_transcript
        modified_segments.append({
            "id": 9999, "text": "This is a new segment added by the test script.",
            "start_time": 9998.0, "end_time": 9999.0
        })
        update_payload = {"segments": modified_segments}
        update_response = requests.put(f"{API_BASE_URL}/meeting/{encoded_request_id}/transcript/plain?username={TEST_USERNAME}", json=update_payload)
        print_response(update_response, "Update Plain Transcript")

        # Verify that the diarized transcript was cleared as a side effect
        print("\n--- Verifying side effects of transcript update ---")
        final_status_res = requests.get(f"{API_BASE_URL}/meeting/{encoded_request_id}/status?username={TEST_USERNAME}").json()
        if final_status_res['data']['diarized_transcript'] is None:
            print("  -> SUCCESS: Diarized transcript was correctly cleared after update.")
        else:
            print("  -> FAILURE: Diarized transcript was NOT cleared after update.")
    else:
        print("  -> SKIPPING transcript update test: No plain transcript was available.")

    # ==================================================================
    # STEP 7: TEST MEETING CANCELLATION
    # ==================================================================
    print("\n\n--- STEP 7: TESTING MEETING CANCELLATION ---")
    cancel_request_id = f"test_cancel_{int(time.time())}"
    encoded_cancel_id = quote(cancel_request_id)
    print(f"Using new requestId for cancellation test: {cancel_request_id}")

    # Start a new meeting
    cancel_start_payload = { 'requestId': cancel_request_id, 'username': TEST_USERNAME, 'language': 'vi', 'filename': "cancel.wav", 'bbhName': "Meeting to be Cancelled", 'Type': "Test", 'Host': "Test" }
    requests.post(f"{API_BASE_URL}/meeting/start-bbh", data=cancel_start_payload)
    
    # Upload one chunk but NOT the last one
    requests.post(f"{API_BASE_URL}/meeting/upload-file-chunk", data={'requestId': cancel_request_id, 'isLastChunk': 'False'}, files={'FileData': (os.path.basename(chunks[0]), open(chunks[0], 'rb'), 'audio/wav')})

    # Test Cancel Endpoint
    print(f"\nTesting: DELETE /meeting/{encoded_cancel_id}/cancel")
    cancel_response = requests.delete(f"{API_BASE_URL}/meeting/{encoded_cancel_id}/cancel?username={TEST_USERNAME}")
    print_response(cancel_response, "Cancel Meeting")

    # Verify that the job is gone and returns a 404
    print(f"\n--- Verifying that cancelled job '{cancel_request_id}' is no longer accessible ---")
    verify_cancel_res = requests.get(f"{API_BASE_URL}/meeting/{encoded_cancel_id}/status?username={TEST_USERNAME}")
    if verify_cancel_res.status_code == 404:
        print(f"  -> SUCCESS: Received 404 Not Found for cancelled job, as expected.")
    else:
        print(f"  -> FAILURE: Expected 404 for cancelled job but got {verify_cancel_res.status_code}.")

    print("\n\n--- ALL TESTS COMPLETE ---")

if __name__ == "__main__":
    run_full_test()
