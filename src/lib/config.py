"""
Module de configuration pour Veritas
Fournit un accès facile à la configuration via des variables globales
"""

import os
import dotenv
from .yaml_config import get_config

# Charger la configuration
config = get_config()

# Configuration exposée pour compatibilité avec le code existant
CREW_API_KEY = config.get("crew", "api_key", default="")
CREW_BASE_URL = config.get("crew", "base_url", default="https://openrouter.ai/api/v1")
CREW_MODEL = config.get("crew", "model", default="openrouter/openai/gpt-4.1-mini")
CREW_TEMPERATURE = config.get("crew", "temperature", default=0.7)
CREW_MAX_TOKENS = config.get("crew", "max_tokens", default=4000)

# Configuration spécifique à Veritas
MIN_SIMILARITY_THRESHOLD = config.get("veritas", "min_similarity_threshold", default=0.75)
BM25_TOP_K = config.get("veritas", "bm25_top_k", default=20)
DEBUG = config.get("veritas", "debug", default=False)
QUERY_EXPANSION = config.get("veritas", "query_expansion", default=True)

# Désactiver la télémétrie
os.environ["CREWAI_TELEMETRY"] = "False"
os.environ["TELEMETRY_ENABLED"] = "False"
os.environ["OPENTELEMETRY_ENABLED"] = "False"

# Fonction pour accéder à la configuration des agents
def get_agent_config(agent_type):
    """
    Récupère la configuration d'un agent spécifique
    
    Args:
        agent_type: Type d'agent (page_selector, sentence_filter, etc.)
        
    Returns:
        Dictionnaire de configuration pour l'agent
    """
    return config.get_agent_config(agent_type)

# Fonction pour accéder aux prompts
def get_prompt_config(prompt_type):
    """
    Récupère la configuration d'un prompt spécifique
    
    Args:
        prompt_type: Type de prompt (page_selector, sentence_filter, etc.)
        
    Returns:
        Dictionnaire de configuration pour le prompt
    """
    return config.get_prompt_config(prompt_type)