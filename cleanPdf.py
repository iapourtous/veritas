#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import List, Dict
from tqdm import tqdm
from pypdf import PdfReader, PdfWriter
import nltk
from nltk.tokenize import sent_tokenize
from crewai import Crew, Process
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO

# Import depuis le projet Veritas
from lib.agents import TextFormatterAgent, AgentFactory
from lib import config

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

class PdfCleaner:
    def __init__(self, verbose: bool = True):
        """
        Initialise le nettoyeur de PDF avec l'agent de formatage
        
        Args:
            verbose: Si True, affiche la progression
        """
        self.verbose = verbose
        self.formatter_agent = TextFormatterAgent.create()
    
    def clean_page(self, page_text: str, page_number: int) -> str:
        """
        Nettoie une page de texte en utilisant l'agent de formatage
        
        Args:
            page_text: Texte de la page à nettoyer
            page_number: Numéro de la page
            
        Returns:
            Texte nettoyé
        """
        # Créer la tâche de formatage
        task = TextFormatterAgent.create_task(
            self.formatter_agent, 
            page_text, 
            page_number
        )
        
        # Créer et exécuter la crew
        crew = Crew(
            agents=[self.formatter_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        # Exécuter la tâche
        result = crew.kickoff()
        return str(result).strip()
    
    def clean_pdf_to_pdf(self, input_path: str, output_path: str) -> bool:
        """
        Lit un PDF, utilise l'agent pour corriger le texte et génère un nouveau PDF
        avec le texte formaté.
        
        Args:
            input_path: Chemin vers le fichier PDF d'entrée
            output_path: Chemin où écrire le nouveau PDF
            
        Returns:
            True si le nettoyage a réussi, False sinon
        """
        if not os.path.exists(input_path):
            print(f"❌ Erreur: Le fichier {input_path} n'existe pas.")
            return False
        
        try:
            # Lire le PDF original
            reader = PdfReader(input_path)
            total_pages = len(reader.pages)
            
            if self.verbose:
                print(f"📄 Traitement de {total_pages} pages du PDF {input_path}...")
            
            # Créer le dossier de sortie si nécessaire
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Liste pour stocker les textes nettoyés de chaque page
            cleaned_pages = []
            
            # Traiter chaque page
            for i in tqdm(range(total_pages), desc="Pages traitées", disable=not self.verbose):
                page = reader.pages[i]
                raw_text = page.extract_text()
                
                if raw_text.strip():
                    # Nettoyer le texte avec l'agent
                    if self.verbose:
                        print(f"  Nettoyage de la page {i+1}/{total_pages}...")
                    
                    start_time = time.time()
                    cleaned_text = self.clean_page(raw_text, i)
                    end_time = time.time()
                    
                    if self.verbose:
                        print(f"  ✓ Page {i+1} nettoyée en {end_time - start_time:.2f} secondes")
                    
                    # Stocker le texte nettoyé
                    cleaned_pages.append(cleaned_text)
                else:
                    # Page vide ou sans texte
                    cleaned_pages.append("")
            
            # Créer un nouveau PDF avec les textes nettoyés
            if self.verbose:
                print(f"📄 Création du nouveau PDF...")
            
            # Créer le document PDF
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Définir les styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            normal_style = styles['Normal']
            
            # Style personnalisé pour le texte principal
            main_style = ParagraphStyle(
                'MainText',
                parent=normal_style,
                fontSize=11,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=12
            )
            
            # Style pour les en-têtes de page
            header_style = ParagraphStyle(
                'Header',
                parent=styles['Heading2'],
                fontSize=14,
                leading=16,
                alignment=TA_LEFT,
                spaceAfter=10,
                spaceBefore=10
            )
            
            # Créer le contenu du document
            content = []
            
            # Titre du document
            content.append(Paragraph(f"Document: {os.path.basename(input_path)}", title_style))
            content.append(Spacer(1, 0.25*inch))
            
            # Ajouter chaque page nettoyée
            for i, page_text in enumerate(cleaned_pages):
                # En-tête de page
                content.append(Paragraph(f"Page {i+1}", header_style))
                
                # Si la page est vide, ajouter un message
                if not page_text:
                    content.append(Paragraph("(Page vide ou sans texte extractible)", main_style))
                    content.append(Spacer(1, 0.1*inch))
                    continue
                
                # Diviser le texte en paragraphes (les lignes vides marquent les paragraphes)
                paragraphs = page_text.split('\n\n')
                
                for para in paragraphs:
                    # Ignorer les paragraphes vides
                    if not para.strip():
                        continue
                    
                    # Ajouter le paragraphe et un petit espace
                    content.append(Paragraph(para.replace('\n', ' '), main_style))
                
                # Ajouter un espace entre les pages
                content.append(Spacer(1, 0.2*inch))
            
            # Générer le PDF
            doc.build(content)
            
            if self.verbose:
                print(f"✅ PDF nettoyé créé: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement du PDF: {str(e)}")
            return False
    
    def extract_sentences(self, input_path: str, output_path: str) -> bool:
        """
        Lit un PDF, utilise l'agent pour nettoyer le texte, puis extrait les phrases
        dans un fichier JSON.
        
        Args:
            input_path: Chemin vers le fichier PDF d'entrée
            output_path: Chemin où écrire le fichier JSON contenant les phrases
            
        Returns:
            True si l'extraction a réussi, False sinon
        """
        if not os.path.exists(input_path):
            print(f"❌ Erreur: Le fichier {input_path} n'existe pas.")
            return False
        
        try:
            # Lire le PDF
            reader = PdfReader(input_path)
            total_pages = len(reader.pages)
            
            if self.verbose:
                print(f"📄 Extraction des phrases de {total_pages} pages du PDF {input_path}...")
            
            # Extraire et corriger chaque page
            all_sentences = []
            sentences_by_page = {}
            
            # Traiter chaque page avec une barre de progression
            for i in tqdm(range(total_pages), desc="Pages traitées", disable=not self.verbose):
                page = reader.pages[i]
                raw_text = page.extract_text()
                
                if raw_text.strip():
                    # Nettoyer le texte avec l'agent
                    if self.verbose:
                        print(f"  Nettoyage de la page {i+1}/{total_pages}...")
                    
                    start_time = time.time()
                    cleaned_text = self.clean_page(raw_text, i)
                    end_time = time.time()
                    
                    if self.verbose:
                        print(f"  ✓ Page {i+1} nettoyée en {end_time - start_time:.2f} secondes")
                    
                    # Découper en phrases
                    sentences = sent_tokenize(cleaned_text)
                    
                    # Filtrer les phrases trop courtes ou vides
                    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                    
                    # Ajouter les phrases avec leur numéro de page
                    sentences_by_page[i] = valid_sentences
                    
                    # Ajouter à la liste globale
                    all_sentences.extend(valid_sentences)
            
            # Écrire le résultat en JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "document": os.path.basename(input_path),
                    "total_pages": total_pages,
                    "total_sentences": len(all_sentences),
                    "sentences": all_sentences,
                    "sentences_by_page": sentences_by_page
                }, f, ensure_ascii=False, indent=2)
            
            if self.verbose:
                print(f"✅ {len(all_sentences)} phrases extraites et écrites dans {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des phrases: {str(e)}")
            return False
    
    def clean_pdf_to_text(self, input_path: str, output_path: str) -> bool:
        """
        Lit un PDF, utilise l'agent pour corriger le texte et écrit une version propre
        dans un fichier texte.
        
        Args:
            input_path: Chemin vers le fichier PDF d'entrée
            output_path: Chemin où écrire le fichier texte propre
            
        Returns:
            True si le nettoyage a réussi, False sinon
        """
        if not os.path.exists(input_path):
            print(f"❌ Erreur: Le fichier {input_path} n'existe pas.")
            return False
        
        try:
            # Lire le PDF
            reader = PdfReader(input_path)
            total_pages = len(reader.pages)
            
            if self.verbose:
                print(f"📄 Traitement de {total_pages} pages du PDF {input_path}...")
            
            # Créer le dossier de sortie si nécessaire
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Extraire et corriger chaque page
            with open(output_path, 'w', encoding='utf-8') as f:
                # Écrire l'en-tête
                f.write(f"# Document: {os.path.basename(input_path)}\n\n")
                
                # Traiter chaque page avec une barre de progression
                for i in tqdm(range(total_pages), desc="Pages traitées", disable=not self.verbose):
                    page = reader.pages[i]
                    raw_text = page.extract_text()
                    
                    if raw_text.strip():
                        # Nettoyer le texte avec l'agent
                        if self.verbose:
                            print(f"  Nettoyage de la page {i+1}/{total_pages}...")
                        
                        start_time = time.time()
                        cleaned_text = self.clean_page(raw_text, i)
                        end_time = time.time()
                        
                        if self.verbose:
                            print(f"  ✓ Page {i+1} nettoyée en {end_time - start_time:.2f} secondes")
                        
                        # Écrire la page nettoyée
                        f.write(f"## Page {i+1}\n\n")
                        f.write(cleaned_text)
                        f.write("\n\n")
            
            if self.verbose:
                print(f"✅ Document nettoyé écrit dans {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement du PDF: {str(e)}")
            return False

def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description="CleanPDF - Nettoie et corrige le texte extrait d'un PDF avec un agent IA")
    parser.add_argument("input", help="Chemin vers le fichier PDF d'entrée")
    parser.add_argument("--output", "-o", help="Chemin de sortie (par défaut: [input]_clean.pdf)")
    parser.add_argument("--format", "-f", choices=["pdf", "text", "json"], default="pdf", 
                      help="Format de sortie: 'pdf' (par défaut), 'text' pour fichier texte, 'json' pour extraction de phrases")
    parser.add_argument("--verbose", "-v", action="store_true", help="Afficher des informations détaillées")
    
    args = parser.parse_args()
    
    # Définir le chemin de sortie s'il n'est pas spécifié
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        if args.format == "json":
            args.output = f"{input_base}_sentences.json"
        elif args.format == "text":
            args.output = f"{input_base}_clean.txt"
        else:  # pdf
            args.output = f"{input_base}_clean.pdf"
    
    # Initialiser le nettoyeur
    cleaner = PdfCleaner(verbose=args.verbose)
    
    # Exécuter la fonction appropriée
    if args.format == "json":
        success = cleaner.extract_sentences(args.input, args.output)
    elif args.format == "text":
        success = cleaner.clean_pdf_to_text(args.input, args.output)
    else:  # pdf
        success = cleaner.clean_pdf_to_pdf(args.input, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())