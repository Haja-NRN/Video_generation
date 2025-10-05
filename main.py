import os
import logging
import fastapi
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


# Constants
# 1️⃣ Base directory du projet
BASE_DIR = settings.BASE_DIR  # ou ajuste selon ton fichier

# 2️⃣ Dossier où seront sauvegardées les vidéos
VIDEO_STATIC_DIR = BASE_DIR / "static" / "videos"
VIDEO_STATIC_DIR.mkdir(parents=True, exist_ok=True)

# 3️⃣ Dossier local pour les modèles Hugging Face
MODEL_DIR = BASE_DIR / "models"  # ton dossier préféré
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 4️⃣ Configuration du modèle Hugging Face
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "THUDM/CogVideoX-2b")
HF_LOCAL_REPO = MODEL_DIR / "CogVideoX-2b"
ALLOW_AUTO_DOWNLOAD = True  # tu autorises le téléchargement automatique
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
DEFAULT_DEVICE="cuda" if torch.cuda.is_available() else "cpu"
VIDEO_CONFIGS = {
        "tiktok": {
            "num_frames": 49,  # CogVideoX génère 49 frames par défaut
            "height": 480,     # Format carré pour TikTok
            "width": 480,
            "fps": 8,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "max_duration": 60  # TikTok max 60 secondes
        },
        "youtube": {
            "num_frames": 49,
            "height": 480,     
            "width": 720,      # Format paysage pour YouTube
            "fps": 8,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "max_duration": 600  # YouTube Shorts max 60s
        }
    }


class VideoGenerationResponse(BaseModel):
    status: Literal["success", "error"]
    platform: Literal["tiktok", "youtube"]
    video_data: Dict
    message: str
    generation_time: float


import asyncio
import torch
import logging
from pathlib import Path
from fastapi import HTTPException
from transformers import BitsAndBytesConfig
from huggingface_hub import snapshot_download, login
from diffusers import CogVideoXPipeline

logger = logging.getLogger(__name__)

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "THUDM/CogVideoX-2b")
HF_LOCAL_REPO = Path(os.getenv("HF_LOCAL_REPO", "models/CogVideoX-2b")).resolve()
ALLOW_AUTO_DOWNLOAD = os.getenv("ALLOW_AUTO_DOWNLOAD", "1") == "1"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class CogVideoXManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._pipeline = None
            obj._is_initialized = False
            obj._lock = asyncio.Lock()
            cls._instance = obj
        return cls._instance

    async def ensure_initialized(self):
        """Assure que le pipeline est prêt sans bloquer en cas d’erreur."""
        if self._is_initialized:
            return
        async with self._lock:
            if not self._is_initialized:
                await self._initialize()

    async def _initialize(self):
        logger.info("🚀 Initialisation de CogVideoX-2b (offline-first)")

        # Vérifie si le modèle est disponible localement
        if not HF_LOCAL_REPO.exists() or not any(HF_LOCAL_REPO.iterdir()):
            if not ALLOW_AUTO_DOWNLOAD:
                logger.warning(
                    f"⚠️ Modèle absent: {HF_LOCAL_REPO}. Téléchargement automatique désactivé."
                )
                return  # Pas de blocage
            try:
                logger.info(f"📥 Téléchargement du modèle {HF_MODEL_ID} depuis Hugging Face...")
                if HUGGINGFACE_TOKEN:
                    login(token=HUGGINGFACE_TOKEN)
                snapshot_download(
                    repo_id=HF_MODEL_ID,
                    repo_type="model",
                    local_dir=str(HF_LOCAL_REPO),
                    local_dir_use_symlinks=False,
                    revision="main",
                )
                logger.info(f"✅ Modèle téléchargé dans {HF_LOCAL_REPO}")
            except Exception as e:
                logger.error(f"❌ Échec du téléchargement du modèle: {e}")
                return  # Ne bloque pas le serveur

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # ✅ Configuration quantization int8
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            logger.info("🧠 Chargement du modèle (int8, device_map=auto)")
            self._pipeline = CogVideoXPipeline.from_pretrained(
                str(HF_LOCAL_REPO),
                device_map="auto",
                quantization_config=quant_config,
                local_files_only=True  # Ne télécharge rien ici
            )

            # ⚙️ Optimisations GPU
            if device == "cuda":
                try:
                    self._pipeline.enable_model_cpu_offload()
                except Exception:
                    pass
                try:
                    self._pipeline.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass
            else:
                logger.warning("⚠️ GPU non disponible — fonctionnement CPU (lent)")

            self._is_initialized = True
            logger.info("✅ CogVideoX-2b prêt (int8).")

        except Exception as e:
            logger.error(f"❌ Erreur lors de l’initialisation CogVideoX: {e}")
            self._is_initialized = False
            self._pipeline = None
            # ⚠️ On ne lève pas d’exception HTTP ici — pour éviter de bloquer le serveur

    def get_pipeline(self):
        """Retourne le pipeline si disponible, sinon None (au lieu d’erreur)."""
        if not self._is_initialized or self._pipeline is None:
            logger.warning("⚠️ CogVideoX non initialisé ou pipeline absent.")
            return None
        return self._pipeline

    @staticmethod
    def cleanup_memory():
        """Nettoie la mémoire GPU et CPU."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            import gc
            gc.collect()

video_manager = CogVideoXManager()


async def generate_video_with_cogvideox(prompt: str, platform: str, quality: str = "medium", use_negative_prompt: bool = True) -> Dict:
    await video_manager.ensure_initialized()
    
    start_time = datetime.now()
    try:
        config = VIDEO_CONFIGS[platform]
        quality_multipliers = {"low": {"steps": 0.5, "resolution": 0.75}, "medium": {"steps": 1.0, "resolution": 1.0}, "high": {"steps": 1.5, "resolution": 1.0}}
        mult = quality_multipliers[quality]
        num_inference_steps = int(config["num_inference_steps"] * mult["steps"])
        height = int(config["height"] * mult["resolution"])
        width = int(config["width"] * mult["resolution"])

        enhanced_prompt = enhance_prompt(prompt, platform)
        negative = get_negative_prompt(platform) if use_negative_prompt else None

        logger.info(
            f"🎬 Génération vidéo CogVideoX: platform={platform} quality={quality} steps={num_inference_steps} res={width}x{height} prompt={enhanced_prompt[:100]}..."
        )

        pipeline = video_manager.get_pipeline()
        with torch.no_grad():
            video = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=config["num_frames"],
                height=height,
                width=width,
                guidance_scale=config["guidance_scale"],
                generator=torch.Generator(device=DEFAULT_DEVICE).manual_seed(42),
            ).frames[0]

        timestamp = now_configured().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{platform}_{quality}_{timestamp}_{unique_id}.mp4"
        filepath = VIDEO_STATIC_DIR / filename

        export_to_video(video, str(filepath), fps=config["fps"])
        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        video_manager.cleanup_memory()
        generation_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "local_path": f"/static/videos/{filename}",
            "filename": filename,
            "file_size_mb": round(file_size_mb, 2),
            "platform": platform,
            "quality": quality,
            "resolution": f"{width}x{height}",
            "fps": config["fps"],
            "duration_seconds": config["num_frames"] / config["fps"],
            "prompt_original": prompt,
            "prompt_enhanced": enhanced_prompt,
            "generation_time": generation_time,
            "generated_at": now_configured().isoformat(),
        }

    except torch.cuda.OutOfMemoryError:
        logger.error("❌ Mémoire GPU insuffisante")
        video_manager.cleanup_memory()
        raise HTTPException(status_code=507, detail="Mémoire GPU insuffisante. Essayez avec une qualité inférieure.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération: {e}")
        video_manager.cleanup_memory()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la vidéo: {str(e)}")



@app.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video_only(
    platform: Literal["tiktok", "youtube"] = Form(...),
    theme_general: str = Form(...),
    theme_hebdo: str = Form(""),
    texte_inspiration: str = Form(""),
):
    
    try:
        if HUGGINGFACE_TOKEN is None:
            logger.warning("HUGGINGFACE_TOKEN non configuré (si le repo est gated, la génération échouera)")

        parts = [theme_general.strip()]
        if theme_hebdo.strip():
            parts.append(theme_hebdo.strip())
        if texte_inspiration.strip():
            parts.append(texte_inspiration.strip()[:200])
        video_prompt = " - ".join(parts)

        logger.info(f"🎥 Requête génération: platform={platform}, prompt={video_prompt[:140]}")
        result = await generate_video_with_cogvideox(video_prompt, platform)

        return VideoGenerationResponse(
            status="success",
            platform=platform,
            video_data={
                "video_url": result["video_url"],
                "local_path": result["local_path"],
                # "imgbb_success": result["imgbb_success"],
                # "imgbb_url": result["video_url"],
                # "imgbb_delete_url": result.get("imgbb_delete_url"),
                # "imgbb_id": result.get("imgbb_id"),
                "generation_info": {
                    "platform": platform,
                    "theme_general": theme_general,
                    "theme_hebdo": theme_hebdo,
                    "prompt": video_prompt,
                    "generated_at": now_configured().isoformat(),
                },
            },
            message=f"Vidéo générée avec succès pour {platform}",
            generation_time=0.0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération vidéo: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de vidéo: {e}")

