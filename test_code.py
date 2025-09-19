# test_api.py (Comprehensive E2E Test Suite)
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


WEBSOCKET_DONE = threading.Event()

# --- HELPER FUNCTIONS ---

def print_response(response, step_name):
    """Helper to print API response details beautifully."""
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
        if status in ['completed', 'failed', 'cancelled']:
            WEBSOCKET_DONE.set()
            ws.close()
    def on_error(ws, error): print(f"[WebSocket] Error: {error}"); WEBSOCKET_DONE.set()
    def on_close(ws, c, m): print("[WebSocket] Connection closed.")
    def on_open(ws): print("[WebSocket] Connection opened.")
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()


def run_full_test():
    metadata = read_speaker_metadata()
    meetings = group_meeting_chunks()
    enrolled_speakers = {} # To store user_ad for later tests

    # ==================================================================
    # STEP 1: ENROLL ALL SPEAKERS
    # ==================================================================
    print("\n--- STEP 1: ENROLLING ALL SPEAKERS ---")
    speaker_folders = [d for d in os.listdir(SPEAKER_AUDIO_DIR) if os.path.isdir(os.path.join(SPEAKER_AUDIO_DIR, d))]
    for folder_name in speaker_folders:
        meta = metadata.get(folder_name)
        if not meta or not meta.get('user_ad'):
            print(f"  -> WARNING: Skipping folder '{folder_name}' due to missing metadata or user_ad.")
            continue

        user_ad = meta['user_ad']
        audio_files = find_audio_files(os.path.join(SPEAKER_AUDIO_DIR, folder_name))
        if not audio_files:
            print(f"  -> WARNING: Skipping folder '{folder_name}' as it contains no audio files.")
            continue

        files_to_upload_paths = audio_files[:MAX_SAMPLES_PER_SPEAKER]
        files_to_upload_tuples = [('files', (os.path.basename(f), open(f, 'rb'))) for f in files_to_upload_paths]
        payload = {'metadata': json.dumps({"display_name": meta.get('display_name', folder_name), "user_ad": user_ad})}
        
        print(f"\nEnrolling '{user_ad}' ({folder_name}) with {len(files_to_upload_paths)} samples...")
        try:
            response = requests.post(f"{API_BASE_URL}/speaker/", data=payload, files=files_to_upload_tuples, timeout=UPLOAD_TIMEOUT_SECONDS)
            print_response(response, "Enroll Speaker")
            if response.status_code == 201:
                enrolled_speakers[user_ad] = {"folder": folder_name}
        except RequestException as e:
            print(f"  -> ERROR: Connection error during enrollment: {e}")
        time.sleep(1)

    # ==================================================================
    # STEP 2: TEST ALL SPEAKER MANAGEMENT (CRUD) ENDPOINTS
    # ==================================================================
    print("\n\n--- STEP 2: TESTING ALL SPEAKER CRUD ENDPOINTS ---")
    # GET /speaker/ (List All)
    print("\nTesting: GET /speaker/")
    list_response = requests.get(f"{API_BASE_URL}/speaker/")
    print_response(list_response, "List All Speakers")

    speakers_from_api = []

    if list_response.status_code == 200:
            speakers_from_api = list_response.json().get("data", [])
    if speakers_from_api:
        test_user_ad = speakers_from_api[0]['user_ad']

        # GET /speaker/{user_ad} (Get Details)
        print(f"\nTesting: GET /speaker/{test_user_ad}")
        details_response = requests.get(f"{API_BASE_URL}/speaker/{test_user_ad}")
        print_response(details_response, "Get Speaker Details")
        
        # PUT /speaker/{user_ad}/metadata (Update Metadata)
        print(f"\nTesting: PUT /speaker/{test_user_ad}/metadata")
        update_payload = {"display_name": "Takeosimashi"}
        update_response = requests.put(f"{API_BASE_URL}/speaker/{test_user_ad}/metadata", json=update_payload)
        print_response(update_response, "Update Metadata")

        details_response = requests.get(f"{API_BASE_URL}/speaker/{test_user_ad}")
        print_response(details_response, "Get Speaker Upodated Details")

        # POST /speaker/{user_ad}/samples (Add More Samples)
        print(f"\nTesting: POST /speaker/{test_user_ad}/samples")
        folder_to_add_from = 'TV HDQT Koji Iriguchi'
        audio_to_add = find_audio_files(os.path.join(SPEAKER_AUDIO_DIR, folder_to_add_from))
        if len(audio_to_add) > 1:
            sample_file = [('files', (os.path.basename(audio_to_add[-1]), open(audio_to_add[-1], 'rb')))]
            add_sample_response = requests.post(f"{API_BASE_URL}/speaker/{test_user_ad}/samples", files=sample_file, timeout=UPLOAD_TIMEOUT_SECONDS)
            print_response(add_sample_response, "Add Sample")
        else:
            print("  -> Skipping Add Sample test: Not enough audio files.")

        # DELETE /speaker/{user_ad} (Delete)
        user_to_delete = 'takeos'
        print(f"\nTesting: DELETE /speaker/{user_to_delete}")
        delete_response = requests.delete(f"{API_BASE_URL}/speaker/{user_to_delete}")
        print_response(delete_response, "Delete Speaker")

    # ==================================================================
    # STEP 3: PROCESS A FULL MEETING
    # ==================================================================
    print("\n\n--- STEP 3: PROCESSING A FULL MEETING ---")
    if not meetings:
        print("No meetings found to process. Exiting.")
        return
        
    meeting_name, chunks = list(meetings.items())[0]
    request_id = f"test_{meeting_name}_{int(time.time())}"
    
    # POST /meeting/start-bbh
    print(f"\nTesting: POST /meeting/start-bbh for requestId '{request_id}'...")
    start_payload = { 'requestId': request_id, 'username': TEST_USERNAME, 'language': 'vi', 'filename': f"{meeting_name}.wav", 'bbhName': "Test BBH Name", 'Type': "Test Type", 'Host': "Test Host" }
    start_response = requests.post(f"{API_BASE_URL}/meeting/start-bbh", data=start_payload)
    print_response(start_response, "Start Meeting")
    
    # POST /meeting/upload-file-chunk
    print(f"\nTesting: POST /meeting/upload-file-chunk...")
    for i, chunk_path in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        chunk_payload = {'requestId': request_id, 'isLastChunk': str(is_last)}
        files = {'FileData': (os.path.basename(chunk_path), open(chunk_path, 'rb'), 'audio/wav')}
        requests.post(f"{API_BASE_URL}/meeting/upload-file-chunk", data=chunk_payload, files=files)
    print("  -> [Upload Chunks] All chunks sent.")

    # ==================================================================
    # STEP 4: REAL-TIME STATUS AND SEQUENTIAL TRIGGERS
    # ==================================================================
    print("\n--- STEP 4: WAITING FOR TRANSCRIPTION VIA WEBSOCKET ---")
    ws_thread = threading.Thread(target=listen_on_websocket, args=(request_id,))
    ws_thread.start()
    WEBSOCKET_DONE.wait(timeout=600) # Increased timeout
    
    # GET /meeting/{request_id}/status
    print(f"\nTesting: GET /meeting/{request_id}/status (after transcription)")
    status_response = requests.get(f"{API_BASE_URL}/meeting/{request_id}/status?username={TEST_USERNAME}")
    print_response(status_response, "Get Status")
    
    # POST /meeting/{request_id}/diarize
    print(f"\nTesting: POST /meeting/{request_id}/diarize")
    diarize_response = requests.post(f"{API_BASE_URL}/meeting/{request_id}/diarize?username={TEST_USERNAME}")
    print_response(diarize_response, "Trigger Diarization")
    
    WEBSOCKET_DONE.clear()
    print("\n--- WAITING FOR DIARIZATION VIA WEBSOCKET ---")
    WEBSOCKET_DONE.wait(timeout=600) # Increased timeout

    # GET /meeting/{request_id}/status (after diarization)
    print(f"\nTesting: GET /meeting/{request_id}/status (after diarization)")
    status_response_final = requests.get(f"{API_BASE_URL}/meeting/{request_id}/status?username={TEST_USERNAME}")
    print_response(status_response_final, "Get Final Status")
    
    # ==================================================================
    # STEP 5: TEST ALL ANALYSIS AND DOWNLOAD ENDPOINTS
    # ==================================================================
    print("\n\n--- STEP 5: TESTING ANALYSIS AND DOWNLOADS ---")
    
    # POST /meeting/{request_id}/summary (topic)
    print(f"\nTesting: POST /meeting/{request_id}/summary (topic)")
    summary_resp_topic = requests.post(f"{API_BASE_URL}/meeting/{request_id}/summary?username={TEST_USERNAME}", json={"summary_type": "topic"})
    print_response(summary_resp_topic, "Generate Topic Summary")
    
    # POST /meeting/{request_id}/summary (speaker)
    print(f"\nTesting: POST /meeting/{request_id}/summary (speaker)")
    summary_resp_speaker = requests.post(f"{API_BASE_URL}/meeting/{request_id}/summary?username={TEST_USERNAME}", json={"summary_type": "speaker"})
    print_response(summary_resp_speaker, "Generate Speaker Summary")

    # POST /meeting/chat
    print(f"\nTesting: POST /meeting/chat")
    chat_payload = {"requestId": request_id, "username": TEST_USERNAME, "message": "What was the main conclusion?"}
    chat_response = requests.post(f"{API_BASE_URL}/meeting/chat", json=chat_payload)
    print_response(chat_response, "Chat")

    # GET /meeting/{request_id}/download/audio
    print(f"\nTesting: GET /meeting/{request_id}/download/audio")
    audio_dl_response = requests.get(f"{API_BASE_URL}/meeting/{request_id}/download/audio?username={TEST_USERNAME}")
    print(f"  -> [Download Audio] Status: {audio_dl_response.status_code}, Received: {len(audio_dl_response.content)} bytes")

    # GET /meeting/{request_id}/download/document
    print(f"\nTesting: GET /meeting/{request_id}/download/document")
    doc_params = {'username': TEST_USERNAME, 'template_type': 'bbh_hdqt'}
    doc_dl_response = requests.get(f"{API_BASE_URL}/meeting/{request_id}/download/document", params=doc_params)
    print(f"  -> [Download Document] Status: {doc_dl_response.status_code}, Received: {len(doc_dl_response.content)} bytes")
    
    ws_thread.join()
    print("\n\n--- ALL TESTS COMPLETE ---")

if __name__ == "__main__":
    run_full_test()
