import os
import dotenv
from dotenv import load_dotenv

# Désactiver la télémétrie
os.environ["CREWAI_TELEMETRY"] = "False"
os.environ["TELEMETRY_ENABLED"] = "False"
os.environ["OPENTELEMETRY_ENABLED"] = "False"

# Charger les variables d'environnement
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Configuration par défaut
CREW_API_KEY = os.getenv("CREW_API_KEY", "")
CREW_BASE_URL = os.getenv("CREW_BASE_URL", "https://openrouter.ai/api/v1")
CREW_MODEL = os.getenv("CREW_MODEL", "openrouter/openai/gpt-4.1-mini")
CREW_TEMPERATURE = float(os.getenv("CREW_TEMPERATURE", "0.7"))
CREW_MAX_TOKENS = int(os.getenv("CREW_MAX_TOKENS", "4000"))

# Configuration spécifique à Veritas
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.75"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "True").lower() in ("true", "1", "yes")