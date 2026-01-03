from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
from huggingface_hub import login, HfFolder
import logging
import os

# Import conditionnel de handle_file
try:
    from gradio_client import handle_file
    HAS_HANDLE_FILE = True
except ImportError:
    HAS_HANDLE_FILE = False
    handle_file = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL du Space (format direct .hf.space)
SPACE_URL = "https://dofbi-galsenai-xtts-v2-wolof-inference.hf.space"

# Token HF - CRITIQUE pour éviter le quota
HF_TOKEN = os.environ.get("HF_TOKEN")

# ⭐ AUTHENTIFICATION EXPLICITE avec huggingface_hub
AUTH_SUCCESS = False
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("✅ Authentification Hugging Face réussie !")
        logger.info(f"🔐 Token configuré (longueur: {len(HF_TOKEN)})")
        AUTH_SUCCESS = True
    except Exception as e:
        logger.error(f"❌ Échec de l'authentification HF : {e}")
        logger.error("⚠️ Le wrapper fonctionnera mais avec quota limité !")
else:
    logger.error("❌ ATTENTION : Aucun token HF trouvé dans HF_TOKEN !")
    logger.error("⚠️ L'API aura un quota GPU très limité !")

app = FastAPI(title="XTTS Wolof Wrapper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "XTTS Wolof Wrapper API",
        "version": "1.0",
        "status": "operational",
        "space_url": SPACE_URL,
        "authenticated": AUTH_SUCCESS,
        "has_token": HF_TOKEN is not None,
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
    return {
        "status": "healthy",
        "space_url": SPACE_URL,
        "authenticated": AUTH_SUCCESS,
        "token_valid": AUTH_SUCCESS
    }

@app.get("/test-space")
def test_space_connection():
    try:
        logger.info(f"🔄 Test de connexion à {SPACE_URL}")
        
        if not AUTH_SUCCESS:
            logger.warning("⚠️ Connexion non authentifiée - quota limité")
        else:
            logger.info("🔐 Connexion authentifiée avec token HF")
        
        client = Client(SPACE_URL)
        logger.info("✅ Connexion au Space réussie")
        
        return {
            "status": "connected",
            "space_url": SPACE_URL,
            "authenticated": AUTH_SUCCESS,
            "message": "Le Space est accessible"
        }
    
    except Exception as e:
        logger.error(f"❌ Erreur de connexion : {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/synthesize")
def synthesize_speech(text: str, audio_reference_url: str = None):
    """
    Génère de l'audio à partir de texte en Wolof
    
    Args:
        text: Le texte à synthétiser en Wolof
        audio_reference_url: URL de l'audio de référence (optionnel)
    
    Returns:
        JSON avec l'URL de l'audio généré
    """
    try:
        # Vérifier l'authentification
        if not AUTH_SUCCESS:
            logger.warning("⚠️ Génération sans authentification - quota limité !")
        
        # Audio de référence par défaut
        if not audio_reference_url:
            audio_reference_url = "https://github.com/Dremer404/AUDIO/raw/refs/heads/main/anta.wav"
        
        logger.info(f"📝 Texte à synthétiser : {text}")
        logger.info(f"🎤 Audio de référence : {audio_reference_url}")
        logger.info(f"🔐 Authentifié : {AUTH_SUCCESS}")
        
        # Créer le client Gradio
        client = Client(SPACE_URL)
        logger.info("✅ Client Gradio créé")
        
        # Appel de l'API avec ou sans handle_file
        try:
            if HAS_HANDLE_FILE:
                logger.info("📦 Utilisation de handle_file (gradio-client 2.0+)")
                result = client.predict(
                    text=text,
                    audio_reference=handle_file(audio_reference_url),
                    api_name="/predict"
                )
            else:
                logger.info("📦 Utilisation de l'URL directe (gradio-client 0.7)")
                result = client.predict(
                    text=text,
                    audio_reference=audio_reference_url,
                    api_name="/predict"
                )
        except Exception as predict_error:
            error_msg = str(predict_error)
            logger.error(f"❌ Erreur lors de la prédiction : {error_msg}")
            
            # Analyser le type d'erreur
            if "GPU quota" in error_msg or "exceeded" in error_msg:
                raise HTTPException(
                    status_code=429,
                    detail="Quota GPU dépassé. Vérifiez que le token HF est valide et actif."
                )
            
            if "401" in error_msg or "Unauthorized" in error_msg or "expired" in error_msg:
                raise HTTPException(
                    status_code=401,
                    detail=f"Erreur d'authentification : {error_msg}"
                )
            
            if "403" in error_msg or "Forbidden" in error_msg:
                raise HTTPException(
                    status_code=403,
                    detail="Accès refusé au Space. Vérifiez les permissions du token."
                )
            
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.info(f"✅ Résultat brut : {result}")
        
        # Construire l'URL complète de l'audio
        audio_url = None
        
        if isinstance(result, str):
            # Cas 1 : Chemin local /tmp/gradio/...
            if result.startswith("/tmp/gradio/") or result.startswith("tmp/gradio/"):
                audio_url = f"{SPACE_URL}/gradio_api/file={result}"
                logger.info(f"🔗 Chemin local converti en URL : {audio_url}")
            
            # Cas 2 : Chemin relatif /xxx
            elif result.startswith("/"):
                audio_url = f"{SPACE_URL}/gradio_api/file={result}"
                logger.info(f"🔗 Chemin relatif converti en URL : {audio_url}")
            
            # Cas 3 : URL complète déjà fournie
            elif result.startswith("http"):
                audio_url = result
                logger.info(f"🔗 URL complète reçue : {audio_url}")
            
            # Cas 4 : Autre format, on essaie de construire l'URL
            else:
                audio_url = f"{SPACE_URL}/gradio_api/file={result}"
                logger.info(f"🔗 URL construite : {audio_url}")
        
        # Cas 5 : Le résultat n'est pas une string
        elif isinstance(result, dict) and 'path' in result:
            audio_url = f"{SPACE_URL}/gradio_api/file={result['path']}"
            logger.info(f"🔗 URL extraite du dictionnaire : {audio_url}")
        
        else:
            # Dernière tentative : convertir en string
            audio_url = str(result)
            logger.warning(f"⚠️ Type inattendu ({type(result)}), conversion en string")
        
        if not audio_url:
            raise ValueError("Impossible d'extraire l'URL de l'audio du résultat")
        
        logger.info(f"🎉 Audio généré avec succès !")
        logger.info(f"🔗 URL finale : {audio_url}")
        
        return {
            "status": "success",
            "audio_url": audio_url,
            "text": text,
            "audio_reference": audio_reference_url,
            "authenticated": AUTH_SUCCESS
        }
    
    except HTTPException:
        # Re-lever les HTTPException
        raise
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Erreur générale : {error_msg}")
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {error_msg}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DU WRAPPER XTTS WOLOF")
    logger.info("=" * 60)
    logger.info(f"🌐 Space URL: {SPACE_URL}")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"📦 Gradio Client: {'2.0+ (handle_file)' if HAS_HANDLE_FILE else '0.7 (URL directe)'}")
    
    if AUTH_SUCCESS:
        logger.info(f"✅ Authentification HF : RÉUSSIE")
        logger.info(f"🔐 Token : ...{HF_TOKEN[-10:] if HF_TOKEN else 'N/A'}")
    else:
        logger.error("❌ Authentification HF : ÉCHOUÉE")
        logger.error("⚠️  L'API fonctionnera avec quota GPU limité !")
    
    logger.info("=" * 60)
    logger.info("📚 Documentation : http://localhost:{}/docs".format(port))
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
