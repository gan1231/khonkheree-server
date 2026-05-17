from fastapi import WebSocket, WebSocketDisconnect
from typing import dict as Dict
import uuid, json

# conversation_id → {user_id: WebSocket}
_rooms: Dict[str, Dict[str, WebSocket]] = {}


async def ws_connect(websocket: WebSocket, conv_id: str, user_id: str):
    await websocket.accept()
    _rooms.setdefault(conv_id, {})[user_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            # broadcast to other participants in the same conversation
            for uid, ws in list(_rooms.get(conv_id, {}).items()):
                if uid != user_id:
                    try:
                        await ws.send_text(json.dumps({"from": user_id, **payload}))
                    except Exception:
                        pass
    except WebSocketDisconnect:
        _rooms.get(conv_id, {}).pop(user_id, None)
