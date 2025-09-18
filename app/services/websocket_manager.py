import logging
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages active WebSocket connections for real-time job status updates.
    """

    def __init__(self):
        """
        Initializes the manager with a dictionary to hold active connections.
        The structure is: { "request_id_1": [WebSocket, WebSocket, ...], "request_id_2": [...] }
        """
        self.active_connections: Dict[str, List[WebSocket]] = {}
        logger.info("WebSocketManager initialized.")

    # async def connect(self, websocket: WebSocket, request_id: str):
    #     """
    #     Accepts a new WebSocket connection and adds it to the tracking dictionary.
    #     """
    #     if request_id not in self.active_connections:
    #         self.active_connections[request_id] = []
    #     self.active_connections[request_id].append(websocket)
    #     logger.info(f"WebSocket connected for request_id '{request_id}'. Total connections for this job: {len(self.active_connections[request_id])}.")

    def disconnect(self, websocket: WebSocket, request_id: str):
        """
        Removes a WebSocket connection from tracking upon disconnection.
        """
        if request_id in self.active_connections:
            # Add a check to prevent errors if the websocket is already gone
            if websocket in self.active_connections[request_id]:
                self.active_connections[request_id].remove(websocket)
            if not self.active_connections[request_id]:
                del self.active_connections[request_id]
            logger.info(f"WebSocket disconnected for request_id '{request_id}'.")

            
    async def broadcast_to_job(self, request_id: str, message: Dict[str, Any]):
        """
        Sends a JSON message to all clients connected for a specific meeting job to push real-time updates to the frontend.
        """
        if request_id in self.active_connections:
            connections = self.active_connections[request_id]
            logger.info(f"Broadcasting message to {len(connections)} client(s) for request_id '{request_id}'.")
            
            send_tasks = [conn.send_json(message) for conn in connections]
            
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Handle clients that may have disconnected abruptly
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to send message to a client for job '{request_id}': {result}. The connection may be closed.")

    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Sends a message to a single, specific WebSocket connection.
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message to a websocket: {e}")

websocket_manager = WebSocketManager()
