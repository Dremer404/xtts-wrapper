from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XTTS Wolof Wrapper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SPACE_URL = "https://dofbi-galsenai-xtts-v2-wolof-inference.hf.space"

# Définir le token HF comme variable d'environnement
# Gradio Client le lira automatiquement
HF_TOKEN = "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"
if HF_TOKEN and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = HF_TOKEN

@app.get("/")
def root():
    return {
        "message": "XTTS Wolof Wrapper API",
        "version": "1.0",
        "status": "operational",
        "authenticated": "HF_TOKEN" in os.environ,
        "endpoints": {
            "GET /": "Informations sur l'API",
            "GET /health": "Vérifie que l'API fonctionne",
            "GET /test-space": "Teste la connexion au Space HF",
            "POST /synthesize": "Génère de l'audio à partir de texte"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """Vérifie que l'API fonctionne"""
    return {
        "status": "healthy",
        "space_url": SPACE_URL,
        "authenticated": "HF_TOKEN" in os.environ
    }

@app.get("/test-space")
def test_space_connection():
    """Teste la connexion au Space Hugging Face"""
    try:
        logger.info(f"🔄 Test de connexion à {SPACE_URL}")
        
        # Le Client lira automatiquement HF_TOKEN depuis l'environnement
        client = Client(SPACE_URL)
        
        if "HF_TOKEN" in os.environ:
            logger.info("✅ Connexion authentifiée (HF_TOKEN présent)")
        else:
            logger.info("⚠️ Connexion non authentifiée (quota limité)")
        
        return {
            "status": "connected",
            "space_url": SPACE_URL,
            "authenticated": "HF_TOKEN" in os.environ,
            "message": "Le Space est accessible"
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur de connexion : {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de se connecter au Space : {str(e)}"
        )

@app.post("/synthesize")
def synthesize_speech(text: str, audio_reference_url: str = None):
    """
    Génère de l'audio à partir de texte en Wolof
    
    Args:
        text: Le texte à synthétiser en Wolof
        audio_reference_url: URL de l'audio de référence (optionnel)
    
    Returns:
        JSON avec l'URL de l'audio généré
    
    Example:
        POST /synthesize?text=Naka nga def&audio_reference_url=https://example.com/audio.wav
    """
    try:
        # Audio de référence par défaut
        if not audio_reference_url:
            audio_reference_url = "https://github.com/Dremer404/AUDIO/raw/refs/heads/main/anta.wav"
        
        logger.info(f"📝 Texte à synthétiser : {text}")
        logger.info(f"🎤 Audio de référence : {audio_reference_url}")
        
        # Le Client lira automatiquement HF_TOKEN depuis l'environnement
        if "HF_TOKEN" in os.environ:
            logger.info("🔐 Connexion authentifiée avec HF_TOKEN")
        else:
            logger.warning("⚠️ Connexion non authentifiée (quota GPU limité)")
        
        client = Client(SPACE_URL)
        
        # Appel de l'API Gradio
        result = client.predict(
            text=text,
            audio_reference=handle_file(audio_reference_url),
            api_name="/predict"
        )
        
        logger.info(f"✅ Audio généré avec succès : {result}")
        
        return {
            "status": "success",
            "audio_url": result,
            "text": text,
            "audio_reference": audio_reference_url
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur lors de la génération : {error_msg}")
        
        # Messages d'erreur personnalisés
        if "GPU quota" in error_msg or "exceeded" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Quota GPU dépassé. Connectez-vous avec un token HF ou attendez la réinitialisation du quota."
            )
        
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération : {error_msg}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    if "HF_TOKEN" in os.environ:
        logger.info(f"🔐 Token HF détecté (authentification activée)")
    else:
        logger.warning("⚠️ Aucun token HF (quota GPU limité)")
    
    logger.info(f"🚀 Démarrage du serveur sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
