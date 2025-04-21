#!/usr/bin/env python3
import os
import argparse
import json
import time
from .core import Veritas
from lib import config

def main():
    """
    Point d'entrée principal de l'application Veritas
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