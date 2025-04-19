#!/usr/bin/env python3
import os
import argparse
import json
from typing import Dict, List, Any
import time
from crewai import Crew, Process
from lib.pdf_parser import PdfParser
from lib.bm25 import BM25Ranker
from lib.levenshtein import align_response, build_factual_response
from lib.agents import VeritasCrewBuilder
from lib import config
from tqdm import tqdm

class Veritas:
    """
    Veritas - Une chaîne d'agents IA qui répond aux questions sur un PDF sans halluciner,
    en utilisant uniquement des phrases extraites du document.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialise Veritas avec un chemin de fichier PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF à analyser
        """
        self.pdf_path = pdf_path
        self.pdf_parser = PdfParser()
        self.bm25_ranker = BM25Ranker()
        self.all_sentences = []
        self.sentences_by_page = {}
        
        # Extraire le texte et les phrases du PDF
        print("📄 Extraction du texte et découpage en phrases...")
        self.all_sentences, self.sentences_by_page = self.pdf_parser.extract_all_sentences(pdf_path)
        print(f"✅ {len(self.all_sentences)} phrases extraites de {len(self.sentences_by_page)} pages.")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Répond à une question en utilisant seulement les phrases du document
        
        Args:
            question: La question à répondre
            
        Returns:
            Un dictionnaire contenant:
            - question: la question posée
            - answer: la réponse alignée sur les phrases du document
            - raw_answer: la réponse générée avant alignement
            - source_sentences: les phrases sources utilisées
            - alignment_details: détails de l'alignement Levenshtein
        """
        print(f"\n📝 QUESTION: {question}")
        
        # Construire et exécuter la crew
        crew_builder = VeritasCrewBuilder(self.pdf_path, question)
        crew = crew_builder.build()
        
        # Étape 1: Sélection des pages pertinentes
        print("\n🧑‍⚖️ Agent 1: Sélection des pages pertinentes...")
        result1 = crew.kickoff()
        
        try:
            # Extraire les pages sélectionnées
            result1_str = str(result1)
            selected_pages = json.loads(result1_str)
            selected_pages_indices = selected_pages.get("selected_pages", [])
            
            if not selected_pages_indices:
                return {
                    "question": question,
                    "answer": "Aucune page pertinente n'a été trouvée dans le document.",
                    "raw_answer": "",
                    "source_sentences": [],
                    "alignment_details": [],
                    "source_pages": []
                }
            
            print(f"✅ {len(selected_pages_indices)} pages sélectionnées: {selected_pages_indices}")
            
            # Étape 2: Collecter toutes les phrases des pages sélectionnées
            all_sentences_from_selected_pages = []
            for page_idx in selected_pages_indices:
                if page_idx in self.sentences_by_page:
                    all_sentences_from_selected_pages.extend(self.sentences_by_page[page_idx])
            
            # Étape 3: Filtrage des phrases pertinentes
            print("\n🚫 Agent 2: Filtrage des phrases pertinentes...")
            from lib.agents import SentenceFilterAgent, ResponseGeneratorAgent
            
            # Récréer une nouvelle crew avec toutes les tâches
            sentence_filter = SentenceFilterAgent.create()
            task2 = SentenceFilterAgent.create_task(
                sentence_filter, 
                question, 
                all_sentences_from_selected_pages
            )
            
            new_crew = Crew(
                agents=[sentence_filter],
                tasks=[task2],
                verbose=True,
                process=Process.sequential
            )
            
            result2 = new_crew.kickoff()
            
            # Extraire les phrases sélectionnées avec gestion d'erreur robuste
            result2_str = str(result2)
            # Utiliser une extraction plus robuste pour éviter les erreurs de parsing JSON
            try:
                selected_sentences_data = json.loads(result2_str)
                selected_sentences = selected_sentences_data.get("selected_sentences", [])
            except json.JSONDecodeError:
                # En cas d'erreur de parsing JSON, extraire les phrases directement du texte
                # avec une approche d'extraction de texte simple
                print(f"⚠️ Erreur de parsing JSON, utilisation d'une méthode alternative d'extraction")
                selected_sentences = []
                
                # Chercher les phrases directement dans la réponse
                import re
                # Chercher du texte entre guillemets qui ressemble à des phrases
                matches = re.findall(r'"([^"]+)"', result2_str)
                for match in matches:
                    if len(match) > 20:  # Filtrer les courts extraits qui ne sont probablement pas des phrases
                        selected_sentences.append(match)
            
            if not selected_sentences:
                return {
                    "question": question,
                    "answer": "Aucune phrase pertinente n'a été trouvée dans les pages sélectionnées.",
                    "raw_answer": "",
                    "source_sentences": [],
                    "alignment_details": [],
                    "source_pages": selected_pages_indices
                }
            print(f"✅ {len(selected_sentences)} phrases sélectionnées.")
            
            # Étape 4: Génération de la réponse
            print("\n🗣️ Agent 3: Génération de la réponse...")
            response_generator = ResponseGeneratorAgent.create()
            task3 = ResponseGeneratorAgent.create_task(
                response_generator, 
                question, 
                selected_sentences
            )
            
            final_crew = Crew(
                agents=[response_generator],
                tasks=[task3],
                verbose=True,
                process=Process.sequential
            )
            
            try:
                result3 = final_crew.kickoff()
                raw_answer = str(result3)
            except Exception as e:
                print(f"⚠️ Erreur lors de la génération de la réponse: {str(e)}")
                raw_answer = "Impossible de générer une réponse cohérente à partir des phrases sélectionnées."
            print("✅ Réponse générée.")
            
            # Étape 5: Alignement Levenshtein
            print("\n📐 Alignement avec Levenshtein...")
            alignment_results = align_response(
                raw_answer, 
                selected_sentences, 
                threshold=config.MIN_SIMILARITY_THRESHOLD
            )
            
            # Construire la réponse factuelle
            factual_answer = build_factual_response(alignment_results)
            
            # Résultat
            return {
                "question": question,
                "answer": factual_answer,
                "raw_answer": raw_answer,
                "source_sentences": selected_sentences,
                "alignment_details": alignment_results,
                "source_pages": selected_pages_indices  # Ajout des pages sources
            }
        
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
            return {
                "question": question,
                "answer": f"Une erreur s'est produite lors du traitement: {str(e)}",
                "raw_answer": "",
                "source_sentences": [],
                "alignment_details": [],
                "source_pages": []
            }

def main():
    """
    Point d'entrée principal
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Veritas - Réponses factuelles basées sur PDF")
    parser.add_argument("pdf", help="Chemin vers le fichier PDF")
    parser.add_argument("question", help="Question à poser au document")
    parser.add_argument("--output", "-o", help="Fichier de sortie pour le rapport complet (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    parser.add_argument("--debug", "-d", action="store_true", help="Mode debug (affiche plus d'informations)")
    parser.add_argument("--no-query-expansion", action="store_true", help="Désactiver l'expansion de requête pour BM25")
    
    args = parser.parse_args()
    
    # Vérifier que le fichier PDF existe
    if not os.path.exists(args.pdf):
        print(f"❌ Erreur: Le fichier PDF '{args.pdf}' n'existe pas.")
        return 1
    
    # Configurer le mode debug
    if args.debug:
        os.environ["DEBUG"] = "True"
        config.DEBUG = True
    
    # Configurer l'expansion de requête
    if args.no_query_expansion:
        os.environ["QUERY_EXPANSION"] = "False"
        config.QUERY_EXPANSION = False
    
    # Initialiser Veritas
    start_time = time.time()
    veritas = Veritas(args.pdf)
    
    # Répondre à la question
    result = veritas.answer_question(args.question)
    
    # Afficher la réponse
    print("\n" + "="*80)
    print(f"QUESTION: {result['question']}")
    print("-"*80)
    print(f"RÉPONSE: {result['answer']}")
    
    # Afficher les pages sources
    if 'source_pages' in result and result['source_pages']:
        print("-"*80)
        print("SOURCES:")
        for page_num in result['source_pages']:
            print(f"- Page {page_num+1}")
    print("="*80)
    
    # Afficher les informations supplémentaires en mode verbeux
    if args.verbose:
        print("\nRÉPONSE BRUTE:")
        print(result["raw_answer"])
        
        print("\nPHRASES SOURCES:")
        for i, sentence in enumerate(result["source_sentences"]):
            print(f"{i+1}. {sentence}")
            
        print("\nDÉTAILS D'ALIGNEMENT:")
        for detail in result["alignment_details"]:
            generated_preview = detail['generated'][:100] + "..." if len(detail['generated']) > 100 else detail['generated']
            source_preview = detail['source'][:100] + "..." if detail['source'] and len(detail['source']) > 100 else (detail['source'] or 'NON ALIGNÉE')
            
            print(f"- Générée: {generated_preview}")
            print(f"  Source: {source_preview}")
            print(f"  Similarité: {detail['similarity']:.2f}")
            print(f"  Alignée: {detail['aligned']}")
            print()
    
    # Enregistrer le rapport complet si demandé
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            # Utiliser ensure_ascii=False pour éviter d'échapper les caractères Unicode
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Rapport complet enregistré dans {args.output}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Temps total: {elapsed_time:.2f} secondes")
    
    return 0

if __name__ == "__main__":
    exit(main())