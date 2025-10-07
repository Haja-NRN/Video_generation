from pydantic import BaseModel
from typing import Dict, Literal


class VideoGenerationResponse(BaseModel):
    status: Literal["success", "error"]
    platform: Literal["tiktok", "youtube"]
    video_data: Dict
    message: str
    generation_time: float
