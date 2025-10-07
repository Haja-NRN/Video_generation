import os
from celery import Celery
from kombu import Queue
from config import settings
from celery.schedules import crontab

# Configuration Celery
celery_app = Celery(
    "quarkoia_posts",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["app.celery_app.tasks"]
)

celery_app.conf.update(
    task_routes={
        "app.celery_app.tasks.*": {"queue": "default"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
