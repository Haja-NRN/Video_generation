import os
import logging
import fastapi
from groq import Groq
from huggingface_hub import login, snapshot_download
from config.settings import now_configured, settings
from fastapi import Form, HTTPException
from typing import Dict, Literal
from datetime import datetime
from pydantic import BaseModel
import torch
import uuid
from pathlib import Path
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import asyncio
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
from utils.prompt import enhance_prompt, get_negative_prompt
from app.api.routes import post_generation
from fastapi.staticfiles import StaticFiles
from app.db.database import Base, engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("posts_generation")

# Chargement des variables d'environnement
load_dotenv()

app=fastapi.FastAPI()
# Créer les tables dans la base de données si elles n'existent pas
Base.metadata.create_all(bind=engine)

app.include_router(
    post_generation.router
)

# Monter le dossier pour servir les vidéos
app.mount("/videos", StaticFiles(directory="static/videos"), name="videos")