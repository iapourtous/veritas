import os
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
import json
from . import config
import litellm
from .pdf_parser import PdfParser
from .bm25 import BM25Ranker

class AgentFactory:
    """Fabrique pour créer des agents avec configuration cohérente"""
    
    @staticmethod
    def create_agent(role: str, goal: str, backstory: str) -> Agent:
        """
        Crée un agent avec la configuration définie dans config.py
        
        Args:
            role: Rôle spécifique de l'agent
            goal: Objectif de l'agent
            backstory: Contexte et motivation de l'agent
            
        Returns:
            Un agent configuré
        """
        # Désactiver la télémétrie
        os.environ["CREWAI_TELEMETRY"] = "False"
        
        # Configuration pour OpenRouter
        from crewai import LLM
        
        llm = LLM(
            model=config.CREW_MODEL,
            api_key=config.CREW_API_KEY,
            base_url=config.CREW_BASE_URL,
            max_tokens=config.CREW_MAX_TOKENS,
            temperature=config.CREW_TEMPERATURE
        )
        
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

class PageSelectorAgent:
    """Agent qui sélectionne les pages pertinentes du document"""
    
    @staticmethod
    def create() -> Agent:
        return AgentFactory.create_agent(
            role="Sélectionneur de Pages Expert",
            goal="Identifier les pages les plus pertinentes par rapport à la question posée",
            backstory="Expert en analyse documentaire avec une compréhension approfondie de la pertinence "
                     "contextuelle. Tu excelles à identifier rapidement les sections de document qui "
                     "contiennent des informations vraiment utiles pour répondre à des questions spécifiques."
        )
    
    @staticmethod
    def create_task(agent: Agent, question: str, pages: List[str], top_pages_indices: List[int]) -> Task:
        # Créer un dictionnaire des pages avec leur contenu pour l'affichage
        pages_preview = {}
        for i, page_idx in enumerate(top_pages_indices):
            if page_idx < len(pages):
                page_content = pages[page_idx]
                # Limiter la longueur du contenu à afficher
                preview = page_content[:500] + '...' if len(page_content) > 500 else page_content
                pages_preview[str(page_idx)] = preview
        
        return Task(
            description=f"""
            QUESTION: {question}
            
            Tu dois sélectionner les pages véritablement pertinentes pour répondre à cette question.
            
            Voici les pages présélectionnées par un algorithme de recherche:
            {json.dumps(pages_preview, ensure_ascii=False)}
            
            INSTRUCTIONS CRITIQUES:
            1. Examine attentivement chaque page présélectionnée
            2. Détermine quelles pages contiennent des informations DIRECTEMENT pertinentes pour la question
            3. Recherche spécifiquement:
               - Les pages qui répondent EXPLICITEMENT à la question posée
               - Les pages contenant des informations juridiques ou techniques précises liées à la question
               - Les exemples concrets ou cas spécifiques répondant à la question
            4. ÉVITE de sélectionner des pages qui:
               - Ne mentionnent le sujet que de façon tangentielle ou contextuelle
               - Contiennent des informations trop générales sans répondre précisément
               - Font seulement référence à d'autres sections
            5. Explique tes choix pour chaque page
            6. Retourne UNIQUEMENT les indices des pages pertinentes dans un tableau JSON formaté comme suit:
               {{"selected_pages": [0, 2, 5]}}
            
            IMPORTANT: Sois très sélectif. Il vaut mieux choisir peu de pages vraiment pertinentes que beaucoup de pages partiellement pertinentes.
            La qualité de la réponse finale dépend entièrement de ta sélection.
            """,
            agent=agent,
            expected_output="Un JSON contenant les indices des pages les plus pertinentes sélectionnées."
        )

class SentenceFilterAgent:
    """Agent qui filtre les phrases pertinentes"""
    
    @staticmethod
    def create() -> Agent:
        return AgentFactory.create_agent(
            role="Expert en Filtrage de Phrases",
            goal="Isoler uniquement les phrases réellement pertinentes pour répondre à la question posée",
            backstory="Spécialiste en analyse sémantique avec une capacité exceptionnelle à distinguer "
                     "l'information pertinente du bruit. Tu possèdes un sens aigu du détail et une "
                     "compréhension approfondie de la façon dont l'information contextuelle contribue "
                     "à répondre précisément à une question."
        )
    
    @staticmethod
    def create_task(agent: Agent, question: str, sentences: List[str]) -> Task:
        return Task(
            description=f"""
            QUESTION: {question}
            
            Tu dois sélectionner UNIQUEMENT les phrases qui contiennent des informations DIRECTEMENT pertinentes 
            pour répondre PRÉCISÉMENT à cette question.
            
            Voici toutes les phrases extraites des pages pertinentes:
            {json.dumps({i: sent for i, sent in enumerate(sentences)}, ensure_ascii=False)}
            
            INSTRUCTIONS STRICTES:
            1. Analyse chaque phrase individuellement par rapport à la question
            2. Interprète la question de manière LITTÉRALE et cherche les phrases qui y répondent EXPLICITEMENT
            3. Pour cette question "{question}":
               - Identifie les phrases qui mentionnent PRÉCISÉMENT le sujet demandé 
               - Priorise les phrases qui contiennent des RÉPONSES CONCRÈTES (chiffres, durées, règles spécifiques)
               - Cherche des EXEMPLES ou CAS PRATIQUES qui illustrent directement la réponse
            
            4. Sélectionne UNIQUEMENT les phrases qui:
               - Répondent DIRECTEMENT à la question posée
               - Contiennent des INFORMATIONS FACTUELLES et PRÉCISES liées à la question
               - Apportent une VALEUR AJOUTÉE claire à la réponse
            
            5. ÉVITE DE SÉLECTIONNER les phrases qui:
               - Ne mentionnent le sujet que de façon tangentielle
               - Sont purement contextuelles ou introductives
               - Ne contiennent que des informations très générales
               - Font référence à d'autres sections sans apporter d'information concrète
            
            6. Retourne directement les phrases pertinentes sélectionnées, dans leur forme originale complète, 
               dans un tableau JSON (et non leurs indices). Format:
               {{"selected_sentences": ["Phrase complète 1", "Phrase complète 2", "Phrase complète 3"]}}
            
            Sois EXTRÊMEMENT sélectif - il vaut mieux choisir 2-3 phrases parfaitement pertinentes que 10 phrases partiellement pertinentes.
            
            ATTENTION:
            - Pour une question factuelle, cherche une RÉPONSE FACTUELLE précise
            - Pour une question sur une définition, cherche une DÉFINITION explicite
            - Pour une question sur des règles, cherche l'ÉNONCÉ EXACT des règles
            
            IMPORTANT: N'altère JAMAIS le texte original des phrases. Conserve-les exactement telles qu'elles apparaissent.
            """,
            agent=agent,
            expected_output="Un JSON contenant UNIQUEMENT les phrases directement pertinentes pour répondre à la question posée."
        )

class ResponseGeneratorAgent:
    """Agent qui génère une réponse basée uniquement sur les phrases sélectionnées"""
    
    @staticmethod
    def create() -> Agent:
        return AgentFactory.create_agent(
            role="Rédacteur de Réponses Factuelles",
            goal="Créer une réponse précise et factuelle basée uniquement sur les phrases fournies",
            backstory="Expert en communication factuelle avec un talent pour synthétiser l'information "
                     "de manière claire et précise. Tu es réputé pour ta rigueur et ton engagement à "
                     "ne jamais introduire d'informations non vérifiées ou d'hallucinations dans tes réponses."
        )
    
    @staticmethod
    def create_task(agent: Agent, question: str, selected_sentences: List[str]) -> Task:
        return Task(
            description=f"""
            QUESTION: {question}
            
            Tu dois générer une réponse EXCLUSIVEMENT basée sur les phrases validées suivantes:
            {json.dumps(selected_sentences, ensure_ascii=False)}
            
            INSTRUCTIONS STRICTES ET CRITIQUES:
            1. Utilise UNIQUEMENT les phrases fournies telles quelles, mot pour mot
            2. Idéalement, COPIE-COLLE les phrases exactes sans AUCUNE modification
            3. N'altère PAS les phrases, même pour améliorer la grammaire ou la cohérence
            4. Limite tes interventions à:
               - Ajouter des conjonctions simples entre les phrases (et, car, mais...)
               - Ordonner les phrases de façon logique
            5. Ne rajoute ABSOLUMENT AUCUNE information extérieure
            6. Ne fais AUCUNE supposition ou extrapolation
            7. Si les phrases ne contiennent pas assez d'informations, dis simplement que
               tu ne peux pas répondre complètement en te basant sur les données disponibles
            
            TRÈS IMPORTANT:
            - Le système vérifiera que chaque partie de ta réponse correspond EXACTEMENT aux phrases originales
            - La précision de ces correspondances est cruciale pour éviter les hallucinations
            - Ne reformule PAS ou ne paraphrase PAS les phrases, même légèrement
            - Préfère assembler les phrases originales même si le résultat est moins fluide
            
            Ta réponse DOIT être vérifiable en la comparant mot pour mot aux phrases fournies.
            """,
            agent=agent,
            expected_output="Une réponse factuelle composée uniquement des phrases originales fournies, assemblées avec une intervention minimale."
        )

class TextFormatterAgent:
    """Agent qui corrige les erreurs de formatage du texte extrait des PDFs"""
    
    @staticmethod
    def create() -> Agent:
        return AgentFactory.create_agent(
            role="Expert en Correction de Texte",
            goal="Corriger les erreurs de formatage du texte extrait des PDFs",
            backstory="Spécialiste en linguistique et en traitement de texte avec une expérience "
                     "approfondie dans la correction des problèmes spécifiques à l'extraction de texte "
                     "à partir de documents PDF. Tu excelles dans la détection et la résolution "
                     "des problèmes de séparation incorrecte des mots, de caractères spéciaux "
                     "et d'autres artefacts liés à l'extraction de PDF."
        )
    
    @staticmethod
    def create_task(agent: Agent, page_text: str, page_number: int) -> Task:
        return Task(
            description=f"""
            Tu es chargé de corriger les erreurs de formatage du texte extrait de la page {page_number+1} d'un PDF.
            
            Voici le texte brut avec potentiellement des erreurs de formatage:
            
            {page_text}
            
            INSTRUCTIONS:
            1. Corrige les mots incorrectement séparés par des espaces (ex: "traitem ent" -> "traitement")
            2. Rétablis les espaces corrects autour de la ponctuation
            3. Corrige les caractères spéciaux mal encodés
            4. Préserve toutes les informations originales du texte
            5. Ne modifie pas le contenu ou le sens
            6. Ne rajoute aucune nouvelle information
            7. Ne supprime aucune information existante
            
            Retourne le texte corrigé.
            """,
            agent=agent,
            expected_output="Le texte de la page avec les erreurs de formatage corrigées."
        )

class QueryExpansionAgent:
    """Agent qui transforme une question en une pseudo-réponse pour améliorer la recherche BM25"""
    
    @staticmethod
    def create() -> Agent:
        return AgentFactory.create_agent(
            role="Expert en Expansion de Requêtes",
            goal="Transformer une question simple en une pseudo-réponse riche pour améliorer la recherche documentaire",
            backstory="Spécialiste en recherche d'information avec une capacité à anticiper "
                     "les termes, concepts et contextes pertinents qui pourraient apparaître "
                     "dans une réponse. Tu excelles à identifier les mots-clés, synonymes "
                     "et formulations alternatives pour enrichir une requête simple."
        )
    
    @staticmethod
    def create_task(agent: Agent, question: str) -> Task:
        return Task(
            description=f"""
            QUESTION ORIGINALE: {question}
            
            Tu dois transformer cette question en une pseudo-réponse enrichie qui sera utilisée
            pour améliorer la recherche documentaire dans un PDF avec l'algorithme BM25.
            
            INSTRUCTIONS:
            1. Imagine comment pourrait être formulée une réponse idéale à cette question
            2. Inclus des mots-clés potentiellement pertinents, des synonymes et des concepts associés
            3. Formule ta réponse comme un paragraphe de 3-5 phrases qui couvre les aspects importants
            4. N'invente PAS de faits spécifiques, reste général
            5. Utilise un vocabulaire riche et varié qui pourrait correspondre au document
            6. Inclus les termes techniques appropriés au domaine de la question
            
            Ta pseudo-réponse sera utilisée pour rechercher des passages pertinents dans un document,
            pas pour être présentée à l'utilisateur. L'objectif est d'avoir un texte riche en termes
            pertinents pour améliorer la recherche.
            """,
            agent=agent,
            expected_output="Une pseudo-réponse enrichie pour améliorer la recherche BM25."
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
        sentence_filter = SentenceFilterAgent.create()
        response_generator = ResponseGeneratorAgent.create()
        
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