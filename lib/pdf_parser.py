import os
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import ftfy
from unidecode import unidecode

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

def clean_text(text: str, preserve_accents: bool = True) -> str:
    """
    Nettoyage basique d'un texte. 
    Le nettoyage avancé sera réalisé par l'agent TextFormatterAgent.
    
    Args:
        text: Texte à nettoyer
        preserve_accents: Si True, préserve les accents (recommandé pour le français)
        
    Returns:
        Texte nettoyé
    """
    # Utiliser ftfy pour réparer l'encodage et les caractères mal formés
    text = ftfy.fix_text(text)
    
    # Normaliser les espaces et supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text)  # Remplacer les séquences d'espaces par un seul espace
    
    # Si conservation des accents demandée (pour le français)
    if preserve_accents:
        # Remplacements de base pour améliorer la lisibilité
        replacements = {
            '\u00a0': ' ',  # espace insécable
            '\u202f': ' '   # espace fine insécable
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
    else:
        # Convertir en ASCII simple sans accent (pour l'anglais)
        text = unidecode(text)
    
    return text.strip()

class PdfParser:
    def __init__(self):
        pass
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """
        Extrait le texte d'un PDF page par page
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Une liste de chaînes de caractères, chaque élément contenant le texte d'une page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF {pdf_path} n'existe pas")
        
        pdf_pages = []
        reader = PdfReader(pdf_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Ne conserver que les pages avec du texte
                # Nettoyer le texte dès l'extraction
                clean_page_text = clean_text(text)
                pdf_pages.append(clean_page_text)
        
        return pdf_pages
    
    def split_pages_into_sentences(self, pages: List[str]) -> Dict[int, List[str]]:
        """
        Découpe chaque page en phrases
        
        Args:
            pages: Liste des textes de chaque page (déjà nettoyés et formatés)
            
        Returns:
            Un dictionnaire avec les numéros de page comme clés et les listes de phrases comme valeurs
        """
        sentences_by_page = {}
        
        for page_num, page_text in enumerate(pages):
            sentences = sent_tokenize(page_text)
            
            # Filtrer les phrases vides ou trop courtes et nettoyer les bords
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Éliminer les doublons potentiels (phrases identiques) sur une même page
            unique_sentences = []
            for s in sentences:
                if s not in unique_sentences:
                    unique_sentences.append(s)
            
            sentences_by_page[page_num] = unique_sentences
        
        return sentences_by_page
    
    def extract_all_sentences(self, pdf_path: str) -> Tuple[List[str], Dict[int, List[str]]]:
        """
        Extrait toutes les phrases d'un PDF
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Un tuple contenant:
            - Une liste de toutes les phrases
            - Un dictionnaire avec les phrases par page
        """
        pages = self.extract_text_by_page(pdf_path)
        sentences_by_page = self.split_pages_into_sentences(pages)
        
        # Créer une liste plate de toutes les phrases
        all_sentences = []
        for page_num, sentences in sentences_by_page.items():
            all_sentences.extend(sentences)
        
        return all_sentences, sentences_by_page