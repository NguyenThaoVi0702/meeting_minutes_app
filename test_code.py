import requests
import os
import time
import glob
import json
import threading
import websocket
from requests.exceptions import JSONDecodeError
from urllib.parse import quote

# --- CONFIGURATION ---
API_BASE_URL = "http://api:8072/api/v1"
MEETING_AUDIO_DIR = "meeting_to" 
TEST_USERNAME = "test_user_01"
DOWNLOAD_DIR = "downloads"
REQUEST_TIMEOUT = 120 # Timeout for API requests in seconds
WEBSOCKET_TIMEOUT = 600 # Max time to wait for a websocket event in seconds

# --- HELPER FUNCTIONS ---
def print_separator(title):
    print("\n" + "="*80)
    print(f"===== {title.upper()} =====")
    print("="*80 + "\n")

def print_response(response, step_name):
    print(f"  -> [{step_name}] Status: {response.status_code}")
    try:
        # Limit long summary content for cleaner logs
        json_data = response.json()
        if "summary_content" in json_data and len(json_data["summary_content"]) > 250:
            json_data["summary_content"] = json_data["summary_content"][:250] + "..."
        if "data" in json_data and "summary_content" in json_data["data"] and len(json_data["data"]["summary_content"]) > 250:
             json_data["data"]["summary_content"] = json_data["data"]["summary_content"][:250] + "..."
        
        print(f"     Response: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
    except JSONDecodeError:
        print(f"     Response (non-JSON): {response.text[:500]}...")

def group_meeting_chunks():
    meetings = {}
    search_path = os.path.join(MEETING_AUDIO_DIR, '*.wav')
    for chunk_path in glob.glob(search_path):
        filename = os.path.basename(chunk_path)
        # Assumes format like 'meeting_0.wav', 'meeting_1.wav'
        try:
            meeting_name = "_".join(filename.split('_')[:-1])
            chunk_num = int(filename.split('_')[-1].split('.')[0])
            if meeting_name not in meetings:
                meetings[meeting_name] = []
            meetings[meeting_name].append((chunk_num, chunk_path))
        except (ValueError, IndexError):
            print(f"Warning: Skipping file with unexpected format: {filename}")
            continue

    for name, chunks in meetings.items():
        chunks.sort() # Sort by chunk number
        meetings[name] = [path for _, path in chunks] # Keep only the path

    print(f"Found {len(meetings)} meetings to process in '{MEETING_AUDIO_DIR}'.")
    return meetings

# --- WEBSOCKET CLIENT CLASS ---
class WebSocketClient:
    """A threaded WebSocket client to listen for status updates without blocking."""
    def __init__(self, request_id: str):
        encoded_id = quote(request_id)
        ws_url = f"{API_BASE_URL.replace('http', 'ws')}/meeting/ws/{encoded_id}"
        self.ws_app = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.last_message = {}
        self.is_done_event = threading.Event()
        self.thread = threading.Thread(target=self.ws_app.run_forever)

    def on_open(self, ws):
        print("\n[WebSocket] Connection opened.")

    def on_message(self, ws, message):
        data = json.loads(message)
        status = data.get('status')
        print(f"\n[WebSocket] <<<< STATUS UPDATE: {status} >>>>")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        self.last_message = data
        # Signal completion on final states
        if status in ['transcription_complete', 'completed', 'failed', 'cancelled']:
            self.is_done_event.set()

    def on_error(self, ws, error):
        print(f"[WebSocket] Error: {error}")
        self.is_done_event.set() # Don't hang if there's an error

    def on_close(self, ws, close_status_code, close_msg):
        print("[WebSocket] Connection closed.")
        self.is_done_event.set() # Ensure waiting threads are released on close

    def start(self):
        self.thread.start()

    def wait_for_completion(self, timeout=WEBSOCKET_TIMEOUT):
        print(f"\n[Main Thread] Waiting for WebSocket to signal completion (max {timeout}s)...")
        completed = self.is_done_event.wait(timeout=timeout)
        if not completed:
            print("[Main Thread] TIMEOUT waiting for WebSocket signal.")
            self.ws_app.close()
        else:
            print("[Main Thread] WebSocket signal received.")
        self.thread.join(timeout=2)
        return completed

    def reset(self):
        self.is_done_event.clear()


# --- MAIN TEST SCRIPT ---
def run_full_test():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    meetings = group_meeting_chunks()
    if not meetings:
        print(f"No meeting chunk files found in the '{MEETING_AUDIO_DIR}' directory. Aborting test.")
        return

    meeting_name, chunks = list(meetings.items())[0]
    request_id = f"test_{meeting_name}_{int(time.time())}"

    # ==================================================================
    # STEP 1: UPLOAD & PROCESS MEETING (TRANSCRIPTION & DIARIZATION)
    # ==================================================================
    print_separator(f"STEP 1: PROCESSING MEETING '{meeting_name}' (ID: {request_id})")

    # Start WebSocket listener in the background
    ws_client = WebSocketClient(request_id)
    ws_client.start()
    time.sleep(2) # Give the websocket time to connect

    # Start the meeting
    start_payload = {
        'requestId': request_id, 'username': TEST_USERNAME, 'language': 'vi',
        'filename': f"{meeting_name}.wav", 'bbhName': "Cuộc Họp Đánh Giá Dự Án",
        'Type': "Kiểm tra định kỳ", 'Host': "Giám đốc",
        'meetingMembers': json.dumps(["Anh Toàn", "Chị Linh", "Bạn Hùng"])
    }
    print("Testing: POST /meeting/start-bbh")
    response = requests.post(f"{API_BASE_URL}/meeting/start-bbh", data=start_payload, timeout=REQUEST_TIMEOUT)
    print_response(response, "Start Meeting")

    # Upload chunks
    print(f"\nUploading {len(chunks)} audio chunks...")
    for i, chunk_path in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        chunk_payload = {'requestId': request_id, 'isLastChunk': str(is_last)}
        files = {'FileData': (os.path.basename(chunk_path), open(chunk_path, 'rb'), 'audio/wav')}
        requests.post(f"{API_BASE_URL}/meeting/upload-file-chunk", data=chunk_payload, files=files, timeout=REQUEST_TIMEOUT)
        print(f"  - Uploaded chunk {i+1}/{len(chunks)} (isLastChunk={is_last})")

    # Wait for transcription to complete
    ws_client.wait_for_completion()
    if ws_client.last_message.get("status") != "transcription_complete":
        print("ERROR: Transcription did not complete successfully. Aborting further tests.")
        return
    
    ws_client.reset()

    # Trigger and wait for diarization
    print("\nTesting: POST /diarize")
    requests.post(f"{API_BASE_URL}/meeting/{quote(request_id)}/diarize?username={TEST_USERNAME}", timeout=REQUEST_TIMEOUT)
    ws_client.wait_for_completion()
    if ws_client.last_message.get("status") != "completed":
        print("ERROR: Diarization did not complete successfully. Aborting further tests.")
        return

    # ==================================================================
    # STEP 2: GENERATE AND DOWNLOAD ALL SUMMARY TYPES
    # ==================================================================
    print_separator("STEP 2: GENERATING & DOWNLOADING SUMMARIES")
    summary_types_to_test = ['topic', 'speaker', 'summary_bbh_hdqt', 'summary_nghi_quyet']
    
    for s_type in summary_types_to_test:
        # Generate/Get Summary
        print(f"\nTesting: POST /summary (type: {s_type})")
        summary_payload = {"summary_type": s_type}
        summary_response = requests.post(f"{API_BASE_URL}/meeting/{quote(request_id)}/summary?username={TEST_USERNAME}", json=summary_payload, timeout=REQUEST_TIMEOUT)
        print_response(summary_response, f"Generate '{s_type}' Summary")

        # Download Document
        print(f"Testing: GET /download/document (type: {s_type})")
        doc_params = {'username': TEST_USERNAME, 'summary_type': s_type}
        doc_dl_response = requests.get(f"{API_BASE_URL}/meeting/{quote(request_id)}/download/document", params=doc_params, timeout=REQUEST_TIMEOUT)
        if doc_dl_response.status_code == 200:
            save_path = os.path.join(DOWNLOAD_DIR, f"{request_id}_{s_type}.docx")
            with open(save_path, 'wb') as f:
                f.write(doc_dl_response.content)
            print(f"  -> [Download Document] OK. File saved to '{save_path}'")
        else:
            print_response(doc_dl_response, f"Download Document '{s_type}'")
        time.sleep(1) # Small delay between heavy AI calls

    # ==================================================================
    # STEP 3: CHAT SCENARIOS FOR SUMMARY EDITING
    # ==================================================================
    print_separator("STEP 3: TESTING CHAT EDITING SCENARIOS")

    def send_chat_and_print(message, summary_context=None):
        payload = {"requestId": request_id, "username": TEST_USERNAME, "message": message}
        if summary_context:
            payload["summary_type_context"] = summary_context
        print(f"\n>>> Sending Chat (context: {summary_context}): '{message}'")
        response = requests.post(f"{API_BASE_URL}/meeting/chat", json=payload, timeout=REQUEST_TIMEOUT)
        print_response(response, "Chat Response")
        return response.json().get("response", "")

    # --- Scenario 1: Ambiguous edit request ---
    send_chat_and_print("Sửa tóm tắt này cho tôi")

    # --- Scenario 2: Edit a summary that has NOT been generated yet ---
    # (Assuming 'speaker' summary wasn't generated above, but our code does generate it.
    # We will test the logic by trying to edit a non-existent type.)
    send_chat_and_print("Sửa biên bản họp theo công việc cho tôi", summary_context="action_items")

    # --- Scenario 3: Successful Edit ---
    # We already generated 'topic' summary in Step 2.
    edit_instruction = "Sửa lại tóm tắt theo chủ đề, hãy nhấn mạnh rằng kế hoạch marketing cho Quý 4 là ưu tiên hàng đầu."
    send_chat_and_print(edit_instruction, summary_context="topic")

    # --- Scenario 4: Simple Question (no editing) ---
    send_chat_and_print("Ai là người chủ trì cuộc họp này?")
    
    # --- Scenario 5: Verify the edit was saved ---
    print("\nVerifying that the summary edit was saved permanently...")
    summary_payload = {"summary_type": "topic"}
    verify_response = requests.post(f"{API_BASE_URL}/meeting/{quote(request_id)}/summary?username={TEST_USERNAME}", json=summary_payload, timeout=REQUEST_TIMEOUT)
    summary_content = verify_response.json().get("data", {}).get("summary_content", "")
    if "marketing" in summary_content.lower() and "quý 4" in summary_content:
        print("  -> SUCCESS: The new summary content was correctly fetched from the database.")
    else:
        print("  -> FAILURE: The edited summary content was not found.")
    print_response(verify_response, "Verify Saved Edit")


    print_separator("ALL TESTS COMPLETE")

if __name__ == "__main__":
    run_full_test()
