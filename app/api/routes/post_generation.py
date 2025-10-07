import os
import logging
import fastapi
from groq import Groq
from config.settings import now_configured, settings
from fastapi import Form, HTTPException, APIRouter
from typing import Dict, Literal
from datetime import datetime
import torch
import uuid
from pathlib import Path
from app.api.schemas.video_generation import VideoGenerationResponse
import asyncio
from dotenv import load_dotenv
from utils.prompt import enhance_prompt, get_negative_prompt
from bytez import Bytez
import aiohttp
import aiofiles
import requests
import json
from app.celery_app.celery_config import celery_app
from moviepy import VideoFileClip, concatenate_videoclips



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

router = APIRouter(tags=["Video generation"])


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
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", None)
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY", None)
PLACID_API_KEY= os.getenv("PLACID_API_KEY", None)
BYTEZ_MODEL_ID = "Lightricks/LTX-Video-0.9.7-dev"
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

# Download video function
async def download_video(url: str, dest_path: str):
    """Télécharge la vidéo de manière asynchrone."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Erreur téléchargement vidéo: {resp.status}")
            async with aiofiles.open(dest_path, 'wb') as f:
                await f.write(await resp.read())
    return dest_path    

# Upload to IMGBB
def upload_to_imgbb(file_path: str):
    """
    Upload un fichier (image ou vidéo) sur Imgbb et retourne la réponse JSON complète.
    """
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                "https://api.imgbb.com/1/upload",
                params={"key": IMGBB_API_KEY},
                files={"image": f}  # Imgbb attend le champ 'image'
            )
        
        result = response.json()
        return result

    except Exception as e:
        return {"success": False, "error": str(e)}

# Decoupe le prompt en 3 sous-prompts  
# @router.post("/split-prompt")
def split_prompt(prompt: str, duration: int) -> list[str]:
    """
    Découpe le prompt original en plusieurs sous-scènes logiques de 2 secondes chacune.
    Retourne une liste Python de descriptions de scènes, exactement n_scenes éléments.
    """

    SCENE_DURATION = 2
    n_scenes = max(1, duration // SCENE_DURATION)

    system_instruction = (
        "Tu es un réalisateur expert en direction artistique et cinéma. "
        "Ton rôle est de diviser une idée de vidéo en plusieurs plans. "
        "Pour chaque scène, décris une action visuelle cohérente et immersive en 1 ou 2 phrases. "
        "Mentionne la caméra, la lumière, les émotions et l'ambiance. "
        "Sois concis mais évocateur, comme un storyboard de publicité. "
        "La réponse doit être uniquement une liste Python valide contenant des chaînes de texte. "
        "Exemple : [\"Scène 1\", \"Scène 2\", \"Scène 3\"]. Aucun texte supplémentaire."
    )

    user_prompt = (
        f"Prompt global : {prompt}\n"
        f"Durée totale : {duration}s\n"
        f"Découpe exactement {n_scenes} scènes de 2 secondes chacune.\n"
        "Retourne uniquement la liste Python des descriptions de chaque scène."
    )

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content.strip()

    # Nettoyage si jamais le modèle ajoute un formatage imprévu
    if text.startswith("```"):
        text = text.split("```")[1]
    text = text.strip()

    # Conversion en liste Python
    try:
        scenes = json.loads(text)
    except json.JSONDecodeError:
        try:
            scenes = eval(text)
        except Exception:
            scenes = [s.strip() for s in text.split("\n") if s.strip()]

    # ⚡ On force le nombre exact de scènes
    scenes = scenes[:n_scenes]

    # Si le modèle renvoie moins de scènes, on complète par des placeholders
    while len(scenes) < n_scenes:
        scenes.append("Scène vide")

    return scenes

# Génération avec Bytez
async def generate_video_with_bytez(prompt: str, 
                                    platform: str,
                                    duration: int, 
                                    quality: str = "medium", 
                                    use_negative_prompt: bool = False) -> Dict:
    start_time = datetime.now()

    try:
        # ⚙️ Préparation du prompt
        config = VIDEO_CONFIGS[platform]
        enhanced_prompt = enhance_prompt(prompt, platform)
        negative = get_negative_prompt(platform) if use_negative_prompt else None
        
        logger.info(f"🎬 Génération vidéo Bytez: platform={platform}, prompt='{enhanced_prompt[:100]}...'")

        # 🧠 Découpe du prompt global en plusieurs sous-scènes
        scenes = split_prompt(enhanced_prompt, duration)
        logger.info(f"📽️ {len(scenes)} sous-scènes générées pour {duration}s")

        sdk = Bytez(BYTEZ_API_KEY)
        model = sdk.model(BYTEZ_MODEL_ID)

        scene_paths = []
        # 🎥 Génération séquentielle des clips
        for i, scene_prompt in enumerate(scenes, start=1):
            logger.info(f"▶️ Scène {i}/{len(scenes)} : '{scene_prompt[:80]}...'")
            output, error = model.run(scene_prompt)
            if error:
                if i==1:
                    raise Exception(f"Erreur Bytez (scène {i}): {error}")
                else:
                    break  # Stoppe la génération mais conserve les scènes déjà faites

            if not output or not output.endswith(".mp4"):
                raise Exception(f"L'API Bytez n'a pas renvoyé de vidéo valide (scène {i}): {output}")

            filename = f"scene_{i}_{os.path.basename(output)}"
            local_path = os.path.join(VIDEO_STATIC_DIR, filename)
            await download_video(output, local_path)
            scene_paths.append(local_path)

        # 🧩 Assemblage des clips avec MoviePy
        clips = [VideoFileClip(path) for path in scene_paths]
        final_clip = concatenate_videoclips(clips)
        final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_path = os.path.join(VIDEO_STATIC_DIR, final_filename)
        final_clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)

        # 🧹 Nettoyage des clips individuels (optionnel)
        for clip in clips:
            clip.close()
        for path in scene_paths:
            os.remove(path)

        # 📥 Upload final vers Imgbb
        imgbb_result = upload_to_imgbb(final_path)
        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        generation_time = (datetime.now() - start_time).total_seconds()

        if imgbb_result.get("success"):
            logger.info(f"✅ Upload réussi : {imgbb_result['data']['url']}")
        else:
            logger.warning(f"⚠️ Erreur upload Imgbb : {imgbb_result.get('error')}")

        return {
            "success": True,
            "video_url_dev": None,
            "local_path": f"/{final_path}",
            "filename": final_filename,
            "file_size_mb": round(file_size_mb, 2),
            "imgbb_success": imgbb_result.get("success", False),
            "video_url": imgbb_result["data"]["url"] if imgbb_result.get("success") else None,
            "imgbb_delete_url": imgbb_result["data"].get("delete_url") if imgbb_result.get("success") else None,
            "imgbb_id": imgbb_result["data"].get("id") if imgbb_result.get("success") else None,
            "platform": platform,
            "prompt_original": prompt,
            "prompt_enhanced": enhanced_prompt,
            "generation_time": generation_time,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Erreur génération vidéo: {e}")
        return {"success": False, "error": str(e)}


@router.post("/generate-video-bytez", response_model=VideoGenerationResponse)
async def generate_video_only(
    platform: Literal["tiktok", "youtube"] = Form(...),
    theme_general: str = Form(...),
    theme_hebdo: str = Form(""),
    texte_inspiration: str = Form(""),
    duration: int = Form(6),
):
    try:
        if BYTEZ_API_KEY is None:
            logger.warning("BYTEZ_API_KEY non configuré (si le repo est gated, la génération échouera)")

        parts = [theme_general.strip()]
        if theme_hebdo.strip():
            parts.append(theme_hebdo.strip())
        if texte_inspiration.strip():
            parts.append(texte_inspiration.strip()[:200])
        video_prompt = " - ".join(parts)

        logger.info(f"🎥 Requête génération: platform={platform}, prompt={video_prompt[:140]}")
        result = await generate_video_with_bytez(video_prompt, platform,duration=duration)

        return VideoGenerationResponse(
            status="success",
            platform=platform,
            video_data={
                "video_url": result["video_url"],
                "local_path": result["local_path"],
                "imgbb_success": result["imgbb_success"],
                "imgbb_url": result["video_url"],
                "imgbb_delete_url": result.get("imgbb_delete_url"),
                "imgbb_id": result.get("imgbb_id"),
                "generation_info": {
                    "platform": platform,
                    "theme_general": theme_general,
                    "theme_hebdo": theme_hebdo,
                    "prompt": video_prompt,
                    "generated_at": now_configured().isoformat(),
                },
            },
            message=f"Vidéo générée avec succès pour {platform}",
            generation_time=result["generation_time"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération vidéo: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de vidéo: {e}")
    
@router.post("/generate-video-bytez-async", response_model=Dict)
async def generate_video_async(
    platform: str = Form(...),
    theme_general: str = Form(...),
    theme_hebdo: str = Form(""),
    texte_inspiration: str = Form(""),
    duration:int = Form(2)
):
    """
        generate video ansync pour eviter timeout erreur
    """
    try:
        if BYTEZ_API_KEY is None:
            logger.warning("BYTEZ_API_KEY non configuré (si le repo est gated, la génération échouera)")

        parts = [theme_general.strip()]
        if theme_hebdo.strip():
            parts.append(theme_hebdo.strip())
        if texte_inspiration.strip():
            parts.append(texte_inspiration.strip()[:200])
        video_prompt = " - ".join(parts)

        logger.info(f"🎥 Requête génération: platform={platform}, duration={duration},prompt={video_prompt[:140]}")
        task = celery_app.send_task("video.generate", args=[{"platform": platform, "prompt": video_prompt,"duration":duration}])
        return {"status":"accepted","job_id": task.id}
    except Exception as e:
        logger.error(f"❌ Erreur génération vidéo asynchrone: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de vidéo asynchrone: {e}")
    

# Test endpoint
@router.get("/health")
async def health_check():
    start_time = datetime.now()
    print(start_time)
    # 🚀 Initialisation du SDK Bytez
    sdk = Bytez(BYTEZ_API_KEY)
    model = sdk.model("nachikethmurthy666/text-to-video-ms-1.7b")

    # 🎥 Lancement de la génération
    output, error = model.run("A Magician")
    generation_time = (datetime.now() - start_time).total_seconds()
    print(generation_time)
    return {"status": "ok", "timestamp": generation_time,"output":output, "error": error}