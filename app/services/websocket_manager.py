import logging
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages active WebSocket connections for real-time job status updates.

    This class provides a thread-safe, in-memory mechanism to track which
    WebSocket connections belong to which meeting job (identified by request_id).
    It is designed to be instantiated once and used as a singleton across the
    application.
    """

    def __init__(self):
        """
        Initializes the manager with a dictionary to hold active connections.
        The structure is: { "request_id_1": [WebSocket, WebSocket, ...], "request_id_2": [...] }
        """
        self.active_connections: Dict[str, List[WebSocket]] = {}
        logger.info("WebSocketManager initialized.")

    async def connect(self, websocket: WebSocket, request_id: str):
        """
        Accepts a new WebSocket connection and adds it to the tracking dictionary.

        Args:
            websocket (WebSocket): The WebSocket connection object from FastAPI.
            request_id (str): The unique ID of the meeting job this connection is for.
        """
        await websocket.accept()
        if request_id not in self.active_connections:
            self.active_connections[request_id] = []
        self.active_connections[request_id].append(websocket)
        logger.info(f"WebSocket connected for request_id '{request_id}'. Total connections for this job: {len(self.active_connections[request_id])}.")

    def disconnect(self, websocket: WebSocket, request_id: str):
        """
        Removes a WebSocket connection from tracking upon disconnection.

        Args:
            websocket (WebSocket): The disconnected WebSocket object.
            request_id (str): The ID of the job the connection was for.
        """
        if request_id in self.active_connections:
            self.active_connections[request_id].remove(websocket)
            # If no connections are left for this job, clean up the dictionary key
            if not self.active_connections[request_id]:
                del self.active_connections[request_id]
            logger.info(f"WebSocket disconnected for request_id '{request_id}'.")
        else:
            logger.warning(f"Attempted to disconnect a non-tracked WebSocket for request_id '{request_id}'.")

    async def broadcast_to_job(self, request_id: str, message: Dict[str, Any]):
        """
        Sends a JSON message to all clients connected for a specific meeting job.

        This is the primary method used by the application (e.g., the Redis listener)
        to push real-time updates to the frontend.

        Args:
            request_id (str): The ID of the job to which the message should be sent.
            message (Dict[str, Any]): The JSON-serializable message to send.
        """
        if request_id in self.active_connections:
            connections = self.active_connections[request_id]
            logger.info(f"Broadcasting message to {len(connections)} client(s) for request_id '{request_id}'.")
            
            # Create a list of tasks to send messages concurrently
            # This is more efficient than sending them one by one.
            send_tasks = [conn.send_json(message) for conn in connections]
            
            # Execute all send tasks. The results will be a list of Nones if successful.
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Optional: Handle clients that may have disconnected abruptly
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to send message to a client for job '{request_id}': {result}. The connection may be closed.")
                    # In a more complex system, you might want to trigger a disconnect here.

    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Sends a message to a single, specific WebSocket connection.
        Useful for sending an initial status upon connection.
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message to a websocket: {e}")

# Create a single, shared instance of the manager for the application.
# This instance will be imported into the main.py and api routes.
websocket_manager = WebSocketManager()