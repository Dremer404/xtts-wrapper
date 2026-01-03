from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from gradio_client import Client
from huggingface_hub import login
import logging
import os
import requests
from pathlib import Path
import tempfile

# Import conditionnel pour compatibilité
try:
    from huggingface_hub import HfFolder
    HAS_HFFOLDER = True
except ImportError:
    HAS_HFFOLDER = False

try:
    from gradio_client import handle_file
    HAS_HANDLE_FILE = True
except ImportError:
    HAS_HANDLE_FILE = False
    handle_file = None

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

# URL du Space
SPACE_URL = "https://dofbi-galsenai-xtts-v2-wolof-inference.hf.space"

# ⭐ TOKEN HUGGING FACE DIRECTEMENT DANS LE CODE
# Remplace par ton token (commence par hf_)
HF_TOKEN = "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"  # ← METS TON TOKEN ICI

# Fallback vers variable d'environnement si le token n'est pas défini
if HF_TOKEN == "TON_TOKEN_ICI":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        logger.info("🔑 Token chargé depuis variable d'environnement")
else:
    logger.info("🔑 Token chargé depuis le code")

AUTH_SUCCESS = False

if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        if HAS_HFFOLDER:
            saved_token = HfFolder.get_token()
            if saved_token:
                logger.info("✅ Authentification Hugging Face réussie !")
                logger.info(f"🔐 Token configuré et vérifié (longueur: {len(HF_TOKEN)})")
                AUTH_SUCCESS = True
            else:
                logger.error("❌ Token non enregistré malgré login()")
        else:
            logger.info("✅ Authentification Hugging Face réussie !")
            logger.info(f"🔐 Token configuré (longueur: {len(HF_TOKEN)})")
            AUTH_SUCCESS = True
            
    except Exception as e:
        logger.error(f"❌ Échec de l'authentification HF : {e}")
        logger.error("⚠️ Le wrapper fonctionnera mais avec quota limité !")
else:
    logger.warning("⚠️ Aucun token HF fourni - quota GPU limité")

logger.info("=" * 60)
logger.info("🚀 DÉMARRAGE DU WRAPPER XTTS WOLOF")
logger.info("=" * 60)
logger.info(f"🌐 Space URL: {SPACE_URL}")
logger.info(f"🔌 Port: {os.environ.get('PORT', 8000)}")
logger.info(f"📦 Gradio Client: {'2.0+ (handle_file)' if HAS_HANDLE_FILE else '0.7 (URL directe)'}")
logger.info(f"🔐 Authentification HF : {'✅ ACTIVE' if AUTH_SUCCESS else '❌ INACTIVE'}")
if not AUTH_SUCCESS:
    logger.warning("⚠️  L'API fonctionnera avec quota GPU limité !")
logger.info("=" * 60)
logger.info(f"📚 Documentation : http://localhost:{os.environ.get('PORT', 8000)}/docs")
logger.info("=" * 60)

def download_audio_from_hf(audio_url: str, token: str = None) -> bytes:
    """
    Télécharge l'audio depuis Hugging Face Space
    
    Args:
        audio_url: URL de l'audio sur HF Space
        token: Token HF pour l'authentification
    
    Returns:
        bytes: Contenu de l'audio en bytes
    """
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        logger.info(f"📥 Téléchargement de l'audio depuis : {audio_url}")
        
        response = requests.get(audio_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        logger.info(f"✅ Audio téléchargé ({len(response.content)} bytes)")
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement : {e}")
        raise

@app.get("/")
def root():
    """Informations sur l'API"""
    return {
        "message": "XTTS Wolof Wrapper API",
        "version": "1.0",
        "status": "operational",
        "space_url": SPACE_URL,
        "authenticated": AUTH_SUCCESS,
        "token_present": HF_TOKEN is not None,
        "token_valid": AUTH_SUCCESS,
        "endpoints": {
            "GET /": "Informations sur l'API",
            "GET /health": "Vérifie que l'API fonctionne",
            "GET /test-space": "Teste la connexion au Space HF",
            "POST /synthesize": "Génère de l'audio à partir de texte (retourne l'URL)",
            "POST /synthesize-download": "Génère de l'audio et le retourne directement"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "space_url": SPACE_URL,
        "authenticated": AUTH_SUCCESS,
        "token_present": HF_TOKEN is not None
    }

@app.get("/test-space")
def test_space_connection():
    """Teste la connexion au Space HF avec authentification"""
    try:
        logger.info(f"🔄 Test de connexion à {SPACE_URL}")
        logger.info(f"🔐 Authentifié : {AUTH_SUCCESS}")
        
        client = Client(SPACE_URL)
        if AUTH_SUCCESS:
            logger.info("✅ Client créé - token HF actif via login()")
        else:
            logger.warning("⚠️ Client créé SANS token - quota limité")
        
        logger.info("✅ Connexion réussie au Space")
        
        return {
            "status": "connected",
            "space_url": SPACE_URL,
            "authenticated": AUTH_SUCCESS,
            "message": "Le Space est accessible avec votre compte" if AUTH_SUCCESS else "Le Space est accessible (quota limité)"
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
    Génère de l'audio à partir de texte en wolof
    Retourne l'URL de l'audio sur HF Space
    
    Args:
        text: Texte en wolof à synthétiser
        audio_reference_url: URL de l'audio de référence pour le clonage de voix
    """
    try:
        if not audio_reference_url:
            audio_reference_url = "https://github.com/Dremer404/AUDIO/raw/refs/heads/main/anta.wav"
        
        logger.info(f"📝 Texte à synthétiser : {text}")
        logger.info(f"🎤 Audio de référence : {audio_reference_url}")
        logger.info(f"🔐 Authentifié : {AUTH_SUCCESS}")
        
        client = Client(SPACE_URL)
        if AUTH_SUCCESS:
            logger.info("✅ Client Gradio créé - token HF actif")
        else:
            logger.warning("⚠️ Client créé sans token - quota GPU limité")
        
        if HAS_HANDLE_FILE:
            logger.info("📦 Utilisation de handle_file (gradio-client 2.0+)")
            result = client.predict(
                text=text,
                audio_reference=handle_file(audio_reference_url),
                api_name="/predict"
            )
        else:
            logger.info("📦 Utilisation d'URL directe (gradio-client 0.7)")
            result = client.predict(
                text=text,
                audio_reference=audio_reference_url,
                api_name="/predict"
            )
        
        logger.info(f"✅ Résultat brut : {result}")
        
        # Conversion du chemin local en URL complète
        if isinstance(result, str):
            if result.startswith("/tmp/gradio/") or result.startswith("tmp/gradio/"):
                audio_url = f"{SPACE_URL}/gradio_api/file={result}"
                logger.info(f"🔗 Chemin local converti en URL : {audio_url}")
            elif result.startswith("/"):
                audio_url = f"{SPACE_URL}/gradio_api/file={result}"
                logger.info(f"🔗 Chemin absolu converti en URL : {audio_url}")
            else:
                audio_url = result
                logger.info(f"🔗 URL directe utilisée : {audio_url}")
        else:
            audio_url = result
        
        logger.info(f"🎉 Audio généré avec succès !")
        
        return {
            "status": "success",
            "audio_url": audio_url,
            "download_url": f"/download?url={audio_url}",
            "text": text,
            "audio_reference": audio_reference_url,
            "authenticated": AUTH_SUCCESS
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur lors de la synthèse : {error_msg}")
        
        if "GPU quota" in error_msg or "exceeded" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Quota GPU dépassé. Attendez quelques minutes ou utilisez un token HF valide."
            )
        elif "401" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Problème d'authentification HF. Vérifiez votre token."
            )
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/synthesize-download")
def synthesize_speech_download(text: str, audio_reference_url: str = None):
    """
    Génère de l'audio à partir de texte en wolof
    Télécharge et retourne directement le fichier audio
    
    Args:
        text: Texte en wolof à synthétiser
        audio_reference_url: URL de l'audio de référence pour le clonage de voix
    
    Returns:
        Fichier audio WAV directement téléchargeable
    """
    try:
        # Génère l'audio (utilise la fonction synthesize_speech)
        result = synthesize_speech(text, audio_reference_url)
        audio_url = result["audio_url"]
        
        # Télécharge l'audio depuis HF Space
        audio_content = download_audio_from_hf(audio_url, HF_TOKEN)
        
        # Retourne l'audio comme fichier téléchargeable
        return StreamingResponse(
            iter([audio_content]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=audio_{text[:20]}.wav"
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download")
def download_audio(url: str):
    """
    Télécharge un fichier audio depuis une URL HF Space
    
    Args:
        url: URL complète du fichier audio sur HF Space
    
    Returns:
        Fichier audio directement téléchargeable
    """
    try:
        logger.info(f"📥 Demande de téléchargement : {url}")
        
        # Vérifie que l'URL est valide
        if not url.startswith(SPACE_URL):
            raise HTTPException(
                status_code=400,
                detail="URL invalide - doit provenir du Space HF"
            )
        
        # Télécharge l'audio
        audio_content = download_audio_from_hf(url, HF_TOKEN)
        
        # Retourne l'audio
        return StreamingResponse(
            iter([audio_content]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=audio.wav"
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur de téléchargement : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
