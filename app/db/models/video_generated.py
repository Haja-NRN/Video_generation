from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from datetime import datetime
from app.db.database import Base

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)             # Nom du fichier
    url = Column(String)                                           # URL publique (imgbb ou locale)
    local_path = Column(String)                                     # Chemin local sur le serveur
    platform = Column(String)                                       # "tiktok" ou "youtube"
    theme_general = Column(Text)                                    # Thème général
    theme_hebdo = Column(Text, nullable=True)                       # Thème hebdo (optionnel)
    prompt = Column(Text)                                           # Prompt complet utilisé
    generation_time = Column(Float)                                  # Temps de génération en secondes
    created_at = Column(DateTime, default=datetime.utcnow)          # Date de création
