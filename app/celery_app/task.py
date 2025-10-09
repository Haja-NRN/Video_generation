import asyncio
from .celery_config import celery_app
import time
from app.api.routes.post_generation import generate_video_with_bytez,generate_video_general

@celery_app.task(name="video.generate")
def generate_video_task(payload: dict):
    """
    Salalalalaala
    """
    loop = asyncio.new_event_loop()
    print("New event loop created")
    asyncio.set_event_loop(loop)
    if True:
        platform = payload["platform"]
        prompt = payload["prompt"]
        duration=payload["duration"]
    result = loop.run_until_complete(generate_video_general(prompt=prompt, platform=platform,duration=duration))
    return result