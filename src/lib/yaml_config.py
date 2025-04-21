"""
Module de gestion de configuration YAML pour Veritas.
Charge et fusionne les configurations depuis les fichiers YAML et les variables d'environnement.
"""

import os
import yaml
import dotenv
from typing import Dict, Any, Optional, Union

class YAMLConfig:
    """Gestionnaire de configuration basé sur YAML avec support pour les variables d'environnement"""
    
    def __init__(self, config_dir: str = None):
        """
        Initialise le gestionnaire de configuration
        
        Args:
            config_dir: Répertoire des fichiers de configuration (optionnel)
        """
        # Désactiver la télémétrie par défaut
        os.environ["CREWAI_TELEMETRY"] = "False"
        os.environ["TELEMETRY_ENABLED"] = "False"
        os.environ["OPENTELEMETRY_ENABLED"] = "False"
        
        # Déterminer le répertoire de configuration
        if config_dir is None:
            # Utiliser le chemin par défaut
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.config_dir = os.path.join(base_dir, 'config')
        else:
            self.config_dir = config_dir
            
        # Charger les variables d'environnement avec gestion d'erreur
        env_path = os.path.join(self.config_dir, '.env')
        if os.path.exists(env_path):
            dotenv.load_dotenv(env_path)
            print(f"Configuration chargée depuis {env_path}")
        else:
            print(f"Fichier .env non trouvé à {env_path}, utilisation des valeurs par défaut")
        
        # Initialiser les dictionnaires de configuration
        self.config = {}
        self.agents = {}
        self.prompts = {}
        
        # Charger les configurations YAML
        self._load_yaml_config()
        
        # Appliquer les variables d'environnement (priorité plus élevée)
        self._apply_env_vars()
    
    def _load_yaml_config(self):
        """Charge les configurations depuis les fichiers YAML"""
        yaml_dir = os.path.join(self.config_dir, 'yaml')
        if not os.path.exists(yaml_dir):
            raise FileNotFoundError(f"Le répertoire de configuration YAML n'existe pas: {yaml_dir}")
        
        # Charger d'abord le fichier de configuration par défaut
        defaults_path = os.path.join(yaml_dir, 'defaults.yaml')
        if os.path.exists(defaults_path):
            with open(defaults_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        
        # Charger la configuration des agents
        agents_path = os.path.join(yaml_dir, 'agents.yaml')
        if os.path.exists(agents_path):
            with open(agents_path, 'r', encoding='utf-8') as f:
                self.agents = yaml.safe_load(f) or {}
        
        # Charger la configuration des prompts
        prompts_path = os.path.join(yaml_dir, 'prompts.yaml')
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f) or {}
    
    def _apply_env_vars(self):
        """Applique les variables d'environnement à la configuration"""
        # Mappage des variables d'environnement vers les clés de configuration
        env_mapping = {
            "CREW_API_KEY": ["crew", "api_key"],
            "CREW_BASE_URL": ["crew", "base_url"],
            "CREW_MODEL": ["crew", "model"],
            "CREW_TEMPERATURE": ["crew", "temperature"],
            "CREW_MAX_TOKENS": ["crew", "max_tokens"],
            "MIN_SIMILARITY_THRESHOLD": ["veritas", "min_similarity_threshold"],
            "BM25_TOP_K": ["veritas", "bm25_top_k"],
            "DEBUG": ["veritas", "debug"],
            "QUERY_EXPANSION": ["veritas", "query_expansion"],
        }
        
        # Appliquer les variables d'environnement si elles existent
        for env_var, config_path in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Conversion des types
                if env_var in ["CREW_TEMPERATURE", "MIN_SIMILARITY_THRESHOLD"]:
                    env_value = float(env_value)
                elif env_var in ["CREW_MAX_TOKENS", "BM25_TOP_K"]:
                    env_value = int(env_value)
                elif env_var in ["DEBUG", "QUERY_EXPANSION"]:
                    env_value = env_value.lower() in ("true", "1", "yes")
                
                # Mise à jour de la configuration
                self._set_nested_value(self.config, config_path, env_value)
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any) -> None:
        """
        Définit une valeur dans un dictionnaire imbriqué
        
        Args:
            config: Dictionnaire de configuration
            path: Chemin d'accès (liste de clés)
            value: Valeur à définir
        """
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration par son chemin d'accès
        
        Args:
            *keys: Clés du chemin d'accès
            default: Valeur par défaut si le chemin n'existe pas
            
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_agent_config(self, agent_type: str, default: Any = None) -> Dict[str, Any]:
        """
        Récupère la configuration d'un agent spécifique
        
        Args:
            agent_type: Type d'agent (page_selector, sentence_filter, etc.)
            default: Valeur par défaut si l'agent n'existe pas
            
        Returns:
            Dictionnaire de configuration pour l'agent
        """
        return self.agents.get(agent_type, default or {})
    
    def get_prompt_config(self, prompt_type: str, default: Any = None) -> Dict[str, Any]:
        """
        Récupère la configuration d'un prompt spécifique
        
        Args:
            prompt_type: Type de prompt (page_selector, sentence_filter, etc.)
            default: Valeur par défaut si le prompt n'existe pas
            
        Returns:
            Dictionnaire de configuration pour le prompt
        """
        return self.prompts.get(prompt_type, default or {})
    
    def get_all(self) -> Dict[str, Any]:
        """
        Récupère toute la configuration
        
        Returns:
            Dictionnaire complet de configuration
        """
        combined = self.config.copy()
        combined['agents'] = self.agents
        combined['prompts'] = self.prompts
        return combined

# Instance singleton pour l'accès global
_config_instance = None

def get_config(config_dir: str = None) -> YAMLConfig:
    """
    Récupère l'instance singleton de YAMLConfig
    
    Args:
        config_dir: Répertoire des fichiers de configuration (optionnel)
        
    Returns:
        Instance de YAMLConfig
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = YAMLConfig(config_dir)
    return _config_instance