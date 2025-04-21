"""
Module de définition des agents pour Veritas
"""

import os
import json
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from . import config
import litellm
from .pdf_parser import PdfParser
from .bm25 import BM25Ranker

class AgentFactory:
    """Fabrique pour créer des agents avec configuration cohérente"""
    
    @staticmethod
    def create_agent(agent_type: str) -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Args:
            agent_type: Type d'agent à créer (page_selector, sentence_filter, etc.)
            
        Returns:
            Un agent configuré
        """
        # Désactiver la télémétrie
        os.environ["CREWAI_TELEMETRY"] = "False"
        
        # Obtenir la configuration de l'agent depuis les fichiers YAML
        agent_config = config.get_agent_config(agent_type)
        
        # Configuration pour le LLM
        from crewai import LLM
        
        llm = LLM(
            model=config.CREW_MODEL,
            api_key=config.CREW_API_KEY,
            base_url=config.CREW_BASE_URL,
            max_tokens=config.CREW_MAX_TOKENS,
            temperature=config.CREW_TEMPERATURE
        )
        
        # Créer l'agent avec la configuration
        return Agent(
            role=agent_config.get("role", "Agent Veritas"),
            goal=agent_config.get("goal", "Aider à répondre à des questions sur des documents"),
            backstory=agent_config.get("backstory", "Expert en analyse de documents"),
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

class PageSelectorAgent:
    """Agent qui sélectionne les pages pertinentes du document"""
    
    @staticmethod
    def create() -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Returns:
            Un agent configuré
        """
        return AgentFactory.create_agent("page_selector")
    
    @staticmethod
    def create_task(agent: Agent, question: str, pages: List[str], top_pages_indices: List[int]) -> Task:
        """
        Crée une tâche pour l'agent en utilisant la configuration YAML
        
        Args:
            agent: Agent à qui assigner la tâche
            question: Question posée par l'utilisateur
            pages: Liste des pages du document
            top_pages_indices: Indices des pages pré-sélectionnées par BM25
            
        Returns:
            Une tâche configurée
        """
        # Obtenir le template de prompt depuis la configuration
        prompt_config = config.get_prompt_config("page_selector")
        
        # Créer un dictionnaire des pages avec leur contenu pour l'affichage
        pages_preview = {}
        for i, page_idx in enumerate(top_pages_indices):
            if page_idx < len(pages):
                page_content = pages[page_idx]
                # Limiter la longueur du contenu à afficher
                preview = page_content[:500] + '...' if len(page_content) > 500 else page_content
                pages_preview[str(page_idx)] = preview
        
        # Obtenir le texte du prompt
        task_description_template = prompt_config.get("task_description", "")
        
        # Remplacer manuellement les variables pour éviter les problèmes avec les accolades JSON
        task_description = task_description_template.replace("{question}", question)
        task_description = task_description.replace("{pages_preview}", json.dumps(pages_preview, ensure_ascii=False))
        
        return Task(
            description=task_description,
            agent=agent,
            expected_output=prompt_config.get("expected_output", "")
        )

class SentenceFilterAgent:
    """Agent qui filtre les phrases pertinentes"""
    
    @staticmethod
    def create() -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Returns:
            Un agent configuré
        """
        return AgentFactory.create_agent("sentence_filter")
    
    @staticmethod
    def create_task(agent: Agent, question: str, sentences: List[str]) -> Task:
        """
        Crée une tâche pour l'agent en utilisant la configuration YAML
        
        Args:
            agent: Agent à qui assigner la tâche
            question: Question posée par l'utilisateur
            sentences: Liste des phrases des pages sélectionnées
            
        Returns:
            Une tâche configurée
        """
        # Obtenir le template de prompt depuis la configuration
        prompt_config = config.get_prompt_config("sentence_filter")
        
        # Obtenir le texte du prompt
        task_description_template = prompt_config.get("task_description", "")
        
        # Remplacer manuellement les variables pour éviter les problèmes avec les accolades JSON
        task_description = task_description_template.replace("{question}", question)
        sentences_json = json.dumps({i: sent for i, sent in enumerate(sentences)}, ensure_ascii=False)
        task_description = task_description.replace("{sentences}", sentences_json)
        
        return Task(
            description=task_description,
            agent=agent,
            expected_output=prompt_config.get("expected_output", "")
        )

class ResponseGeneratorAgent:
    """Agent qui génère une réponse basée uniquement sur les phrases sélectionnées"""
    
    @staticmethod
    def create() -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Returns:
            Un agent configuré
        """
        return AgentFactory.create_agent("response_generator")
    
    @staticmethod
    def create_task(agent: Agent, question: str, selected_sentences: List[str]) -> Task:
        """
        Crée une tâche pour l'agent en utilisant la configuration YAML
        
        Args:
            agent: Agent à qui assigner la tâche
            question: Question posée par l'utilisateur
            selected_sentences: Liste des phrases sélectionnées pour répondre
            
        Returns:
            Une tâche configurée
        """
        # Obtenir le template de prompt depuis la configuration
        prompt_config = config.get_prompt_config("response_generator")
        
        # Obtenir le texte du prompt
        task_description_template = prompt_config.get("task_description", "")
        
        # Remplacer manuellement les variables pour éviter les problèmes avec les accolades JSON
        task_description = task_description_template.replace("{question}", question)
        sentences_json = json.dumps(selected_sentences, ensure_ascii=False)
        task_description = task_description.replace("{selected_sentences}", sentences_json)
        
        return Task(
            description=task_description,
            agent=agent,
            expected_output=prompt_config.get("expected_output", "")
        )

class TextFormatterAgent:
    """Agent qui corrige les erreurs de formatage du texte extrait des PDFs"""
    
    @staticmethod
    def create() -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Returns:
            Un agent configuré
        """
        return AgentFactory.create_agent("text_formatter")
    
    @staticmethod
    def create_task(agent: Agent, page_text: str, page_number: int) -> Task:
        """
        Crée une tâche pour l'agent en utilisant la configuration YAML
        
        Args:
            agent: Agent à qui assigner la tâche
            page_text: Texte de la page à corriger
            page_number: Numéro de la page
            
        Returns:
            Une tâche configurée
        """
        # Obtenir le template de prompt depuis la configuration
        prompt_config = config.get_prompt_config("text_formatter")
        
        # Obtenir le texte du prompt
        task_description_template = prompt_config.get("task_description", "")
        
        # Remplacer manuellement les variables pour éviter les problèmes avec les accolades JSON
        task_description = task_description_template.replace("{page_number}", str(page_number))
        task_description = task_description.replace("{page_text}", page_text)
        
        return Task(
            description=task_description,
            agent=agent,
            expected_output=prompt_config.get("expected_output", "")
        )

class QueryExpansionAgent:
    """Agent qui transforme une question en une pseudo-réponse pour améliorer la recherche BM25"""
    
    @staticmethod
    def create() -> Agent:
        """
        Crée un agent en utilisant la configuration YAML
        
        Returns:
            Un agent configuré
        """
        return AgentFactory.create_agent("query_expansion")
    
    @staticmethod
    def create_task(agent: Agent, question: str) -> Task:
        """
        Crée une tâche pour l'agent en utilisant la configuration YAML
        
        Args:
            agent: Agent à qui assigner la tâche
            question: Question à transformer
            
        Returns:
            Une tâche configurée
        """
        # Obtenir le template de prompt depuis la configuration
        prompt_config = config.get_prompt_config("query_expansion")
        
        # Obtenir le texte du prompt
        task_description_template = prompt_config.get("task_description", "")
        
        # Remplacer manuellement les variables pour éviter les problèmes avec les accolades JSON
        task_description = task_description_template.replace("{question}", question)
        
        return Task(
            description=task_description,
            agent=agent,
            expected_output=prompt_config.get("expected_output", "")
        )

class VeritasCrewBuilder:
    """Constructeur pour la crew Veritas"""
    
    def __init__(self, pdf_path: str, question: str):
        self.pdf_path = pdf_path
        self.question = question
        self.pdf_parser = PdfParser()
        self.bm25_ranker = BM25Ranker()
        
    def extract_pages(self) -> List[str]:
        """
        Extrait le texte du PDF sans correction avancée
        (on suppose que le PDF a été pré-traité avec cleanPdf.py)
        
        Returns:
            Liste des pages extraites
        """
        # Extraire les pages
        pages = self.pdf_parser.extract_text_by_page(self.pdf_path)
        print(f"📄 {len(pages)} pages extraites du PDF.")
        return pages
        
    def generate_expanded_query(self, question: str) -> str:
        """
        Utilise l'agent d'expansion de requête pour transformer la question
        en une pseudo-réponse plus riche pour la recherche
        
        Args:
            question: Question originale
            
        Returns:
            Requête augmentée pour améliorer la recherche BM25
        """
        print(f"\n🔍 Expansion de la requête pour améliorer la recherche...")
        
        # Créer l'agent et la tâche
        expansion_agent = QueryExpansionAgent.create()
        task = QueryExpansionAgent.create_task(expansion_agent, question)
        
        # Créer et exécuter la crew
        crew = Crew(
            agents=[expansion_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        # Exécuter la tâche
        result = crew.kickoff()
        expanded_query = str(result)
        
        print(f"✅ Requête enrichie générée.")
        if config.DEBUG:
            print(f"\nQuestion originale : {question}")
            print(f"Requête enrichie : {expanded_query}\n")
            
        return expanded_query
        
    def build(self) -> Crew:
        """
        Construit et retourne la crew Veritas complète
        
        Returns:
            Un objet Crew configuré avec les agents et tâches
        """
        # Extraire le texte du PDF (sans formatage avancé)
        pages = self.extract_pages()
        
        # Déterminer la requête à utiliser pour BM25
        search_query = self.question
        
        # Si l'expansion de requête est activée, générer une requête augmentée
        if config.QUERY_EXPANSION:
            search_query = self.generate_expanded_query(self.question)
        else:
            print(f"\n🔍 Utilisation de la question originale pour la recherche BM25 (expansion désactivée)")
        
        # If we have fewer pages than the BM25 threshold, use all pages
        if len(pages) <= config.BM25_TOP_K:
            top_pages_indices = list(range(len(pages)))
        else:
            # Use BM25 with the expanded query to rank and select top pages
            top_pages_indices = self.bm25_ranker.rank_pages(
                pages, search_query, top_k=config.BM25_TOP_K
            )
        
        # Create agents
        page_selector = PageSelectorAgent.create()
        
        # Create page selection task
        task1 = PageSelectorAgent.create_task(
            page_selector, self.question, pages, top_pages_indices
        )
        
        # Create crew with just the first task
        crew = Crew(
            agents=[page_selector],
            tasks=[task1],
            process=Process.sequential,
            verbose=True
        )
        
        return crew