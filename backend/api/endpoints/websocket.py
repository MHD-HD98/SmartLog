# app/routes/websocket.py

from fastapi import APIRouter, WebSocket
from core.video import get_stream_instance, set_stream_instance, FFmpegStream
from core.config import settings

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()

        if data == "restart":
            stream = get_stream_instance()
            if stream and stream.is_running():
                stream.stop()

            new_stream = FFmpegStream(settings.VIDEO_SOURCE)
            new_stream.start()
            set_stream_instance(new_stream)

            await websocket.send_text("FFmpeg stream restarted successfully.")
        else:
            await websocket.send_text("Unknown command.")
    except Exception as e:
        await websocket.send_text(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# # app/api/endpoints/websocket.py

# from fastapi import APIRouter, WebSocket
# import cv2

# router = APIRouter(tags=["WebSocket"])

# # Shared global video capture from core
# from core.video import cap

# @router.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         data = await websocket.receive_text()
#         if data == "restart":
#             # Reinitialize the capture
#             cap.release()
#             cap.open("21.mp4")
#             await websocket.send_text("Video restarted successfully.")
#         else:
#             await websocket.send_text("Unknown command.")
#     except Exception as e:
#         await websocket.send_text(f"WebSocket error: {e}")
#     finally:
#         await websocket.close()
