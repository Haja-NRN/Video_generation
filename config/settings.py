import os
from pathlib import Path
from configparser import ConfigParser
from typing import List, Optional
from urllib.parse import urlparse
from datetime import datetime
import pytz
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # Informations du projet
    BASE_DIR = BASE_DIR
    PROJECT_NAME: str = "Test API Generation Video"
    PROJECT_DESCRIPTION: str = "API pour la génération de vidéos à partir de prompts textuels"
    PROJECT_VERSION: str = "0.0.1"
    
    # Configuration serveur
    BASE_URL: Optional[str] = None
    
    # Configuration Timezone (sera chargée depuis conf.ini)
    DEFAULT_TIMEZONE: str = "Europe/Paris"  # Valeur par défaut
    _timezone_obj = None
    
    @property
    def TIMEZONE(self):
        """Retourne l'objet timezone configuré"""
        if self._timezone_obj is None:
            try:
                self._timezone_obj = pytz.timezone(self.DEFAULT_TIMEZONE)
            except pytz.UnknownTimeZoneError:
                print(f"❌ Timezone '{self.DEFAULT_TIMEZONE}' invalide, utilisation de 'Europe/Paris'")
                self._timezone_obj = pytz.timezone("Europe/Paris")
                self.DEFAULT_TIMEZONE = "Europe/Paris"
        return self._timezone_obj
    
    # Propriétés calculées à partir de BASE_URL
    @property
    def HOST(self) -> str:
        if self.BASE_URL:
            parsed = urlparse(self.BASE_URL)
            return parsed.hostname or "localhost"
        return "localhost"
    
    @property
    def PORT(self) -> int:
        if self.BASE_URL:
            parsed = urlparse(self.BASE_URL)
            return parsed.port or (443 if parsed.scheme == 'https' else 8008)
        return 8008
    

    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Répertoires
    STATIC_DIR: str = os.getenv("STATIC_DIR", "./static/videos")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./temp")
    
    # Limites et optimisations
    MAX_VIDEO_SIZE_MB: int = 100
    ENABLE_GPU_OPTIMIZATION: bool = True
    DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paramètres vidéo par plateforme
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
    
    # Configuration CORS
    CORS_ORIGINS: List[str] = ["*"]  # À remplacer par vos domaines en production

    def __init__(self):
        self.load_config_from_file()

    def load_config_from_file(self):
        """Charge la configuration uniquement à partir du fichier conf.ini"""
        config = ConfigParser()
        path_conf = BASE_DIR / "config/conf.ini"

        try:
            if Path(path_conf).exists():
                config.read(path_conf)
                print(f"✅ Configuration chargée depuis: {path_conf}")
                
                # Configuration Timezone
                if "TIMEZONE" in config:
                    self.DEFAULT_TIMEZONE = config.get("TIMEZONE", "DEFAULT_TIMEZONE", fallback="Europe/Paris")
                    print(f"✅ Timezone configuré: {self.DEFAULT_TIMEZONE}")
                    # Valider le timezone immédiatement
                    try:
                        pytz.timezone(self.DEFAULT_TIMEZONE)
                    except pytz.UnknownTimeZoneError:
                        print(f"❌ Timezone '{self.DEFAULT_TIMEZONE}' invalide dans conf.ini, utilisation de 'Europe/Paris'")
                        self.DEFAULT_TIMEZONE = "Europe/Paris"
                else:
                    print("⚠️ Section [TIMEZONE] manquante dans conf.ini, utilisation de 'Europe/Paris'")
                
            else:
                print(f"❌ Fichier de configuration non trouvé: {path_conf}")

        except Exception as e:
            print(f"❌ Erreur lors du chargement de la configuration: {e}")


settings = Settings()

# Fonctions utilitaires globales pour les timezones
def now_configured() -> datetime:
    """
    Retourne l'heure actuelle dans le timezone configuré
    Remplace datetime.utcnow() et datetime.now() dans tout le système
    """
    return datetime.now(settings.TIMEZONE)

def to_configured_timezone(dt: datetime) -> datetime:
    """
    Convertit un datetime vers le timezone configuré
    """
    if dt.tzinfo is None:
        # Si naive, assume UTC
        dt = pytz.utc.localize(dt)
    return dt.astimezone(settings.TIMEZONE)

def from_user_input(date_str: str, time_str: str = "00:00") -> datetime:
    """
    Parse une date/heure saisie par l'utilisateur dans le timezone configuré
    """
    try:
        full_str = f"{date_str} {time_str}"
        naive_dt = datetime.strptime(full_str, "%Y-%m-%d %H:%M")
        return settings.TIMEZONE.localize(naive_dt)
    except ValueError as e:
        raise ValueError(f"Format de date invalide: {e}")

def make_timezone_aware(dt: datetime) -> datetime:
    """
    Convertit une datetime naive en timezone aware avec le timezone configuré
    Si la datetime est déjà timezone aware, la retourne telle quelle
    """
    if dt.tzinfo is None:
        return settings.TIMEZONE.localize(dt)
    return dt

def format_configured_time(dt: datetime = None) -> str:
    """
    Formate un datetime en string avec le timezone configuré
    """
    if dt is None:
        dt = now_configured()
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
