# app/api/endpoints/stream.py

from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.video import generate_frames
from database import get_db

router = APIRouter(tags=["Video Stream"])


@router.get("/video_feed")
async def video_feed(request: Request, db: AsyncSession = Depends(get_db)):
    return StreamingResponse(
        generate_frames(request, db),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
