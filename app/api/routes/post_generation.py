import os
import logging
import time
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
from PIL import Image
from abc import ABC, abstractmethod
from runware import Runware, IImageInference,IVideoInference
import base64


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
BASE_URL = "http://127.0.0.1:8000/"
# 1️⃣ Base directory du projet
BASE_DIR = settings.BASE_DIR  # ou ajuste selon ton fichier

# 2️⃣ Dossier où seront sauvegardées les vidéos
VIDEO_STATIC_DIR = BASE_DIR / "static" / "videos"
VIDEO_STATIC_DIR.mkdir(parents=True, exist_ok=True)

# 2️⃣ Dossier où seront sauvegardées les vidéos
IMAGE_STATIC_DIR = BASE_DIR / "static" / "images"
IMAGE_STATIC_DIR.mkdir(parents=True, exist_ok=True)

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





# === Base Class ===
class BaseVideoModel(ABC):
    """Classe abstraite pour tous les modèles de génération vidéo."""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    @abstractmethod
    async def generate(self, prompt: str, context_image: bytes = None) -> str:
        """Retourne l'URL de la vidéo générée."""
        pass


# === Bytez Model ===
class BytezModel(BaseVideoModel):
    """
    Implémentation du modèle Bytez.
    """
    def __init__(self, sdk, model_id):
        super().__init__("bytez")
        self.model = sdk.model(model_id)
    
    async def generate(self, prompt: str, context_image: bytes = None) -> str:
        error = None
        try:
            if BYTEZ_API_KEY is None:
                logger.error("BYTEZ_API_KEY non configuré (si le repo est gated, la génération échouera)")
                raise Exception("BYTEZ_API_KEY non configuré")
            try:
                if context_image:
                    output, error = self.model.run(prompt, context_image=context_image)
                else:
                    output, error = self.model.run(prompt)
            except Exception:
                # Si le modèle Bytez ne supporte pas les images
                output, error = self.model.run(prompt)
            
            if error or not output.endswith(".mp4"):
                raise Exception(error or "Vidéo invalide")
            
            return output , error
        except Exception as e:
            raise Exception(f"Bytez generation error: {e}")


# === Runware Model ===
class RunwareModel(BaseVideoModel):
    """
    Implémentation du modèle Runware.
    Nécessite un client Runware initialisé.
    """
    def __init__(self, client, model_text="klingai:5@3", model_image="klingai:3@2"):
        super().__init__("runware")
        self.client = client
        self.model_text = model_text
        self.model_image = model_image

    async def generate(self, prompt: str, context_image: bytes = None) -> str:
        error= False
        try:
            if context_image:
                # 🖼️ Tentative d'image-to-video
                image_base64 = base64.b64encode(context_image).decode("utf-8")
                frame_images = [
                    {"inputImage": image_base64, "frame": "first"},
                ]
                request = {
                    "taskType": "videoInference",
                    "positivePrompt": f"L'image initiale de la video que tu va generer est l'image que j'ai donnée.{prompt}",
                    "model": self.model_image,
                    "duration": 10,
                    "width": 1280,
                    "height": 720,
                    "frameImages": frame_images,
                    "numberResults": 1
                }
            else:
                # 📝 Text-to-video
                request = {
                    "taskType": "videoInference",
                    "positivePrompt": prompt,
                    "model": self.model_text,
                    "duration": 10,
                    "width": 1280,
                    "height": 720,
                    "numberResults": 1
                }

            response = await self.client.videoInference(requestVideo=request)
            return response

        except Exception as e:
            # 🔁 Si le modèle refuse l’image, on repasse en texte pur
            logger.warning(f"Runware image input failed, retrying as text-only: {e}")
            request = {
                "taskType": "videoInference",
                "positivePrompt": prompt,
                "model": self.model_text,
                "duration": 10,
                "width": 1280,
                "height": 720,
                "numberResults": 1
            }
            response = await self.client.videoInference(requestVideo=request)
            return response

bytez_model=BytezModel(Bytez(BYTEZ_API_KEY), BYTEZ_MODEL_ID)
runware_client = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

# ---- Helpers ---
# Fonction de génération de video generale
# async def generate_video_general(
#                                 prompt: str, 
#                                 duration: int, 
#                                 platform: Literal["tiktok", "youtube"],
#                                 quality: str = "medium", 
#                                 use_negative_prompt: bool = False) -> Dict:
#     """
#     Génère une vidéo en utilisant le service Bytez ou Runware.
#     Utilise la découpe en scènes et l'assemblage final.
#     """
#     start_time = datetime.now()

#     try:
#         model = bytez_model
#         # ⚙️ Préparation du prompt
#         config = VIDEO_CONFIGS[platform]
#         enhanced_prompt = enhance_prompt(prompt, platform)
#         negative = get_negative_prompt(platform) if use_negative_prompt else None
        
#         logger.info(f"🎬 Génération vidéo Bytez: platform={platform}, prompt='{enhanced_prompt[:100]}...'")

#         # 🧠 Découpe du prompt global en plusieurs sous-scènes
#         print("Découpage du prompt en scènes...")
#         scenes = split_prompt(enhanced_prompt, duration)
#         logger.info(f"📽️ {len(scenes)} sous-scènes générées pour {duration}s")


#         scene_paths = []
#         last_frame_path = None  # On garde la dernière image générée

#         # 🎥 Génération séquentielle des clips
#         for i, scene_prompt in enumerate(scenes, start=1):
#             logger.info(f"▶️ Scène {i}/{len(scenes)} : '{scene_prompt[:80]}...'")
            
#             try:
#                 # Si on a une image précédente, on essaie de l'envoyer au modèle
#                 if last_frame_path:
#                     try:
#                         # Ici on prépare l'image (tu peux adapter selon le SDK)
#                         with open(last_frame_path, "rb") as f:
#                             last_frame_bytes = f.read()
                        
#                         # Envoye texte + image (si supporté)
#                         output, error = await model.generate(scene_prompt, context_image=last_frame_bytes)
#                         os.remove(last_frame_path)
#                     except Exception as e:
#                         # Si le modèle ne supporte pas l'image, on continue juste avec le texte
#                         logger.warning(f"⚠️ L'image n'a pas été utilisée pour la scène {i} : {e}")
#                         output, error = await model.generate(scene_prompt)
#                 else:
#                     # Première scène, texte uniquement
#                     output, error = await model.generate(scene_prompt)
                
#                 if error:
#                     if i == 1:
#                         raise Exception(f"Erreur modele : {error}")
#                     else:
#                         break  # Stoppe la génération mais conserve les scènes déjà faites

#                 if not output or not output.endswith(".mp4"):
#                     raise Exception(f"Le modele n'a pas renvoyé de vidéo valide (scène {i}): {output}")

#                 filename = f"scene_{i}_{os.path.basename(output)}"
#                 local_path = os.path.join(VIDEO_STATIC_DIR, filename)
#                 await download_video(output, local_path)
#                 scene_paths.append(local_path)

#                 # On récupère la dernière image de la vidéo générée pour la scène suivante
#                 try:
#                     clip = VideoFileClip(local_path)
#                     last_frame = clip.get_frame(clip.duration - 0.01)  # Dernière frame
#                     last_frame_image = Image.fromarray(last_frame)
#                     last_frame_path = os.path.join(IMAGE_STATIC_DIR, f"last_frame_{i}.png")
#                     last_frame_image.save(last_frame_path)
#                     clip.close()
#                 except Exception as e:
#                     logger.warning(f"⚠️ Impossible d'extraire la dernière frame pour la scène {i}: {e}")
#                     last_frame_path = None

#             except Exception as e:
#                 logger.error(f"❌ Erreur lors de la génération de la scène {i}: {e}")
#                 break

#         if not scene_paths:
#                 raise Exception("Aucune scène générée avec succès")
            
#         # Si une seule scène → on renomme simplement
#         if len(scene_paths) == 1:
#             final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
#             final_path = os.path.join(VIDEO_STATIC_DIR, final_filename)
#             os.rename(scene_paths[0], final_path)
#             logger.info(f"🎞️ Une seule scène générée, renommée en {final_filename}")

#         # Sinon concaténer plusieurs clips
#         else:
#             clips = [VideoFileClip(path) for path in scene_paths]
#             final_clip = concatenate_videoclips(clips)
#             final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
#             final_path = os.path.join(VIDEO_STATIC_DIR, final_filename)
#             final_clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)

#             for clip in clips:
#                 clip.close()
#             for path in scene_paths:
#                 os.remove(path)

#         file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
#         generation_time = (datetime.now() - start_time).total_seconds()

#         return {
#             "success": True,
#             "video_url_dev": None,
#             "local_path": f"/{final_path}",
#             "filename": final_filename,
#             "file_size_mb": round(file_size_mb, 2),
#             "video_url": f"{BASE_URL}videos/{final_filename}",
#             "platform": platform,
#             "prompt_original": prompt,
#             "prompt_enhanced": enhanced_prompt,
#             "generation_time": generation_time,
#             "generated_at": datetime.now().isoformat(),
#         }

#     except Exception as e:
#         logger.error(f"❌ Erreur génération vidéo: {e}")
#         return {"success": False, "error": str(e)}
    
    
    

async def generate_video_general(
    prompt: str,
    duration: int,
    platform: Literal["tiktok", "youtube"],
    quality: str = "medium",
    use_negative_prompt: bool = False,
) -> Dict:
    """
    Génère une vidéo en plusieurs scènes, avec reprise sur erreur.
    Supporte Bytez et garantit un résultat final même si certaines scènes échouent.
    """
    start_time = datetime.now()
    model = bytez_model

    try:
        # ⚙️ Préparation du prompt
        enhanced_prompt = enhance_prompt(prompt, platform)
        negative = get_negative_prompt(platform) if use_negative_prompt else None
        logger.info(f"🎬 Génération vidéo Bytez: platform={platform}, prompt='{enhanced_prompt[:100]}...'")

        # 🧠 Découpage en sous-scènes
        scenes = split_prompt(enhanced_prompt, duration)
        logger.info(f"📽️ {len(scenes)} sous-scènes générées pour {duration}s")

        scene_paths = []
        last_frame_path = None

        # Retry
        MAX_RETRIES = 2  # Réessayer jusqu’à 2 fois si une scène échoue

        for i, scene_prompt in enumerate(scenes, start=1):
            logger.info(f"▶️ Scène {i}/{len(scenes)} : '{scene_prompt[:80]}...'")
            retries = 0

            while retries <= MAX_RETRIES:
                try:
                    # Si une image précédente existe, on la passe comme contexte
                    if last_frame_path and os.path.exists(last_frame_path):
                        with open(last_frame_path, "rb") as f:
                            last_frame_bytes = f.read()
                        output, error = await model.generate(scene_prompt, context_image=last_frame_bytes)
                    else:
                        output, error = await model.generate(scene_prompt)

                    if error:
                        raise Exception(f"Erreur modèle Bytez : {error}")

                    # Vérifie que la sortie est bien une URL et un fichier .mp4
                    if not output or not str(output).endswith(".mp4"):
                        raise Exception(f"Sortie invalide : {output}")

                    # Téléchargement et vérification du fichier local
                    filename = f"scene_{i}_{os.path.basename(output)}"
                    local_path = os.path.join(VIDEO_STATIC_DIR, filename)
                    await download_video(output, local_path)

                    if not os.path.exists(local_path) or os.path.getsize(local_path) < 50000:  # 50 Ko min
                        raise Exception("Vidéo téléchargée invalide ou vide")

                    scene_paths.append(local_path)
                    logger.info(f"✅ Scène {i} enregistrée : {local_path}")

                    # Extraire dernière frame
                    try:
                        clip = VideoFileClip(local_path)
                        last_frame = clip.get_frame(clip.duration - 0.05)
                        last_frame_image = Image.fromarray(last_frame)
                        last_frame_path = os.path.join(IMAGE_STATIC_DIR, f"last_frame_{i}.png")
                        last_frame_image.save(last_frame_path)
                        clip.close()
                    except Exception as e:
                        logger.warning(f"⚠️ Impossible d'extraire la dernière frame pour la scène {i}: {e}")
                        last_frame_path = None

                    break  # ✅ Sort de la boucle retry si succès

                except Exception as e:
                    retries += 1
                    logger.error(f"❌ Erreur lors de la scène {i} (tentative {retries}/{MAX_RETRIES}): {e}")
                    if retries > MAX_RETRIES:
                        logger.error(f"⛔ Scène {i} abandonnée après {MAX_RETRIES} tentatives")
                        break
                    await asyncio.sleep(2)  #  courte pause avant retry

        # Nettoyage des images temporaires
        if last_frame_path and os.path.exists(last_frame_path):
            os.remove(last_frame_path)

        if not scene_paths:
            raise Exception("Aucune scène générée avec succès")

        # 🧩 Fusion ou renommage final
        if len(scene_paths) == 1:
            final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            final_path = os.path.join(VIDEO_STATIC_DIR, final_filename)
            os.rename(scene_paths[0], final_path)
            logger.info(f"🎞️ Une seule scène générée, renommée en {final_filename}")
        else:
            clips = []
            for path in scene_paths:
                try:
                    clips.append(VideoFileClip(path))
                except Exception as e:
                    logger.warning(f"⚠️ Clip corrompu ignoré : {path} ({e})")

            if not clips:
                raise Exception("Aucune scène valide pour la fusion finale")

            final_clip = concatenate_videoclips(clips)
            final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            final_path = os.path.join(VIDEO_STATIC_DIR, final_filename)
            final_clip.write_videofile(final_path, codec="libx264", audio=False, logger=None)

            for clip in clips:
                clip.close()
            for path in scene_paths:
                if os.path.exists(path):
                    os.remove(path)

        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        generation_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "video_url_dev": None,
            "local_path": f"/{final_path}",
            "filename": final_filename,
            "file_size_mb": round(file_size_mb, 2),
            "video_url": f"{BASE_URL}videos/{final_filename}",
            "platform": platform,
            "prompt_original": prompt,
            "prompt_enhanced": enhanced_prompt,
            "generation_time": generation_time,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Erreur génération vidéo: {e}")
        return {"success": False, "error": str(e)}

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
   

# ---- Routes FastAPI ---
@router.post("/generate-video", response_model=VideoGenerationResponse)
async def generate_video_only(
    platform: Literal["tiktok", "youtube"] = Form(...),
    theme_general: str = Form(...),
    theme_hebdo: str = Form(""),
    texte_inspiration: str = Form(""),
    duration: int = Form(2),
):
    try:
        
        parts = []

        # Inclure le platform en premier pour contextualiser le prompt
        parts.append(f"Platform: {platform.capitalize()}")

        # Thèmes et inspiration
        if theme_general.strip():
            parts.append(f"Theme general: {theme_general.strip()}")
        if theme_hebdo.strip():
            parts.append(f"Theme hebdo: {theme_hebdo.strip()}")
        if texte_inspiration.strip():
            # Limiter la longueur de l'inspiration pour éviter trop de texte
            parts.append(f"Inspiration: {texte_inspiration.strip()[:200]}")

        # Combiner en un seul prompt, séparé par " | "
        video_prompt = " | ".join(parts)

        logger.info(f"🎥 Requête génération: platform={platform}, prompt={video_prompt[:140]}")
        result = await generate_video_general(prompt=video_prompt, platform=platform,duration=duration)
        # ✅ AJOUT : gestion du cas d'échec de génération
        if not result.get("success", False):
            logger.error(f"⚠️ Génération échouée : {result.get('error', 'Erreur inconnue')}")
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la génération de la vidéo : {result.get('error', 'Erreur inconnue')}"
            )
        
        return VideoGenerationResponse(
            status="success",
            platform=platform,
            video_data={
                "video_url": result["video_url"],
                "local_path": result["local_path"],
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
    
@router.post("/generate-video-async", response_model=Dict)
async def generate_video_async(
    platform: Literal["tiktok", "youtube"] = Form(...),
    theme_general: str = Form(...),
    theme_hebdo: str = Form(""),
    texte_inspiration: str = Form(""),
    duration:int = Form(2)
):
    """
        generate video async pour eviter timeout erreur
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
    time.sleep(3)
    generation_time = (datetime.now() - start_time).total_seconds()
    print(generation_time)
    return {"status": "ok", "timestamp": generation_time}