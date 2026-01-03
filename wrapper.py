from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from huggingface_hub import login
import logging
import os

# Import conditionnel pour compatibilité anciennes/nouvelles versions
try:
    from huggingface_hub import HfFolder
    HAS_HFFOLDER = True
except ImportError:
    HAS_HFFOLDER = False

# Import conditionnel de handle_file
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

# ⭐ AUTHENTIFICATION HUGGING FACE CRITIQUE
HF_TOKEN = os.environ.get("HF_TOKEN")
AUTH_SUCCESS = False

if HF_TOKEN:
    try:
        # Authentification explicite
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        # Vérification selon la version disponible
        if HAS_HFFOLDER:
            saved_token = HfFolder.get_token()
            if saved_token:
                logger.info("✅ Authentification Hugging Face réussie !")
                logger.info(f"🔐 Token configuré et vérifié (longueur: {len(HF_TOKEN)})")
                AUTH_SUCCESS = True
            else:
                logger.error("❌ Token non enregistré malgré login()")
        else:
            # Pour les nouvelles versions, on fait confiance à login()
            logger.info("✅ Authentification Hugging Face réussie !")
            logger.info(f"🔐 Token configuré (longueur: {len(HF_TOKEN)})")
            AUTH_SUCCESS = True
            
    except Exception as e:
        logger.error(f"❌ Échec de l'authentification HF : {e}")
        logger.error("⚠️ Le wrapper fonctionnera mais avec quota limité !")
else:
    logger.warning("⚠️ Aucun token HF fourni - quota GPU limité")

# Affichage des infos au démarrage
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
            "POST /synthesize": "Génère de l'audio à partir de texte"
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
        
        # ⭐ CRÉATION DU CLIENT (le token vient de HfFolder après login())
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
        
        # ⭐ CRÉATION DU CLIENT (le token est déjà actif via login())
        client = Client(SPACE_URL)
        if AUTH_SUCCESS:
            logger.info("✅ Client Gradio créé - token HF actif")
        else:
            logger.warning("⚠️ Client créé sans token - quota GPU limité")
        
        # Appel avec ou sans handle_file selon la version
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
            "text": text,
            "audio_reference": audio_reference_url,
            "authenticated": AUTH_SUCCESS
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur lors de la synthèse : {error_msg}")
        
        # Gestion des erreurs spécifiques
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
