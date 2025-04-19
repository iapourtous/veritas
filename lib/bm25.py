from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import re
from collections import Counter
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class BM25Ranker:
    """
    Implémentation avancée de BM25 (BM25+) avec analyse sémantique 
    pour le ranking de pages de document
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """
        Initialise le moteur BM25+ avec les paramètres optimaux
        
        Args:
            k1: Paramètre de saturation de fréquence des termes (1.2-2.0 recommandé)
            b: Paramètre de normalisation par la longueur (0.75 recommandé)
            delta: Paramètre BM25+ pour les termes rares (1.0 recommandé)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_freqs = {}  # Fréquence des documents où chaque terme apparaît
        self.idf = {}        # Score IDF pour chaque terme
        self.doc_lens = []   # Longueur de chaque document
        self.avg_doc_len = 0 # Longueur moyenne des documents
        self.total_docs = 0  # Nombre total de documents
        self.corpus_terms = set()  # Ensemble des termes dans le corpus
        
        # Stemmer et stopwords pour le prétraitement
        self.stemmer = SnowballStemmer('french')
        self.stop_words = set(stopwords.words('french'))
        
        # Additions spécifiques pour la conservation des données
        self.domain_terms = {
            # Termes liés à la durée de conservation
            'durée': 3.0, 'conservation': 3.0, 'délai': 3.0, 'stockage': 2.5, 
            'archivage': 2.5, 'effacement': 3.0, 'suppression': 2.5, 'rétention': 3.0,
            'temporaire': 2.0, 'permanent': 2.0, 'période': 2.0, 'temps': 2.0,
            'limité': 2.0, 'illimité': 2.0, 'ans': 2.5, 'mois': 2.5, 'jours': 2.5,
            
            # Termes juridiques RGPD
            'rgpd': 2.0, 'gdpr': 2.0, 'règlement': 1.5, 'protection': 1.5, 
            'donnée': 2.0, 'personnelle': 2.0, 'régulation': 1.5, 'loi': 1.5,
            'article': 1.5, 'paragraphe': 1.2, 'alinéa': 1.2, 'disposition': 1.2,
            
            # Termes liés aux principes du RGPD
            'minimisation': 2.0, 'finalité': 2.0, 'limitation': 2.5, 'proportionnalité': 2.0,
            'nécessaire': 2.0, 'pertinent': 1.5, 'adéquat': 1.5, 'exactitude': 1.5,
            
            # Termes liés aux obligations
            'responsable': 1.5, 'traitement': 1.5, 'obligation': 1.5, 'conformité': 1.5,
            'registre': 2.0, 'documentation': 1.5, 'mesure': 1.5, 'technique': 1.0,
            'organisationnel': 1.0, 'sécurité': 1.0,
            
            # Termes spécifiques aux exceptions ou cas particuliers
            'archive': 2.0, 'statistique': 1.5, 'recherche': 1.5, 'historique': 1.5,
            'scientifique': 1.5, 'intérêt': 1.5, 'public': 1.5, 'légal': 1.5
        }
        
    def _preprocess_text(self, text: str, use_stemming: bool = True) -> List[str]:
        """
        Prétraite le texte en plusieurs étapes avancées
        
        Args:
            text: Texte à prétraiter
            use_stemming: Si True, applique le stemming
            
        Returns:
            Liste de termes prétraités
        """
        # 1. Convertir en minuscules et normaliser les espaces
        text = re.sub(r'\s+', ' ', text.lower())
        
        # 2. Remplacer les caractères spéciaux par des espaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 3. Tokenisation
        tokens = word_tokenize(text, language='french')
        
        # 4. Filtrage des stopwords et mots courts
        tokens = [token for token in tokens if
                 (token not in self.stop_words) and  # Pas un stopword
                 (len(token) > 1) and                # Longueur > 1
                 (not token.isdigit()) and           # Pas juste un chiffre
                 (not all(c in string.punctuation for c in token))]  # Pas juste de la ponctuation
        
        # 5. Stemming (optionnel)
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        # 6. N-grams (2-grams)
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(tokens[i] + '_' + tokens[i+1])
            
        # Combiner tokens et bigrams
        all_terms = tokens + bigrams
            
        return all_terms
    
    def _calculate_idf(self) -> None:
        """Calcule les scores IDF pour tous les termes du corpus"""
        for term in self.corpus_terms:
            doc_freq = self.doc_freqs.get(term, 0)
            # Formule IDF modifiée avec lissage
            self.idf[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            
            # Boosting des termes du domaine
            if term in self.domain_terms:
                self.idf[term] *= self.domain_terms[term]
    
    def _calculate_page_score(self, page_terms: List[str], query_terms: List[str], page_index: int) -> float:
        """
        Calcule le score BM25+ d'une page pour une requête
        
        Args:
            page_terms: Termes prétraités de la page
            query_terms: Termes prétraités de la requête
            page_index: Indice de la page dans le corpus
            
        Returns:
            Score BM25+ de la page
        """
        # Fréquence des termes dans la page
        term_freqs = Counter(page_terms)
        
        # Longueur normalisée de la page
        doc_len = self.doc_lens[page_index]
        norm_doc_len = doc_len / self.avg_doc_len
        
        # Calculer le score BM25+
        score = 0.0
        for term in query_terms:
            if term in self.idf:
                tf = term_freqs.get(term, 0)
                idf = self.idf[term]
                
                # Formule BM25+ avec boost contextuel
                term_score = idf * ((tf * (self.k1 + 1)) / 
                                    (tf + self.k1 * (1 - self.b + self.b * norm_doc_len)) + 
                                    self.delta)
                
                # Augmenter le score pour les termes exacts de la requête
                if term in query_terms:
                    term_score *= 1.2
                
                score += term_score
                
        return score
    
    def _expand_query(self, query_terms: List[str]) -> List[str]:
        """
        Étend la requête avec des termes connexes
        
        Args:
            query_terms: Liste des termes originaux de la requête
            
        Returns:
            Liste enrichie de termes pour la requête
        """
        expanded_terms = query_terms.copy()
        
        # Expansion basée sur des synonymes et termes connexes
        expansion_map = {
            'durée': ['temps', 'période', 'délai'],
            'conservation': ['stockage', 'rétention', 'archivage'],
            'effacement': ['suppression', 'destruction', 'élimination'],
            'rgpd': ['gdpr', 'règlement', 'protection'],
            'limitation': ['restriction', 'bornage', 'plafonnement']
        }
        
        # Ajouter des termes d'expansion
        for term in query_terms:
            # Rechercher des stems similaires au stem du terme
            term_stem = self.stemmer.stem(term) if term in self.domain_terms else term
            
            # Ajouter des termes d'expansion basés sur des correspondances
            for key, expansions in expansion_map.items():
                if term == key or term_stem == self.stemmer.stem(key):
                    for exp_term in expansions:
                        expanded_terms.append(exp_term)
        
        return expanded_terms
    
    def fit(self, pages: List[str]) -> None:
        """
        Prépare le modèle BM25+ sur le corpus de pages
        
        Args:
            pages: Liste des textes de chaque page
        """
        self.total_docs = len(pages)
        self.doc_lens = []
        term_doc_freqs = {}
        
        # Prétraiter tous les documents
        tokenized_pages = []
        for page in pages:
            page_terms = self._preprocess_text(page)
            tokenized_pages.append(page_terms)
            
            # Mettre à jour les statistiques
            self.doc_lens.append(len(page_terms))
            
            # Compter les occurrences de termes uniques par document
            unique_terms = set(page_terms)
            for term in unique_terms:
                term_doc_freqs[term] = term_doc_freqs.get(term, 0) + 1
                self.corpus_terms.add(term)
        
        # Calculer la longueur moyenne des documents
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.total_docs)
        
        # Stocker les fréquences des documents
        self.doc_freqs = term_doc_freqs
        
        # Calculer les scores IDF
        self._calculate_idf()
        
        return tokenized_pages
        
    def rank_pages(self, pages: List[str], query: str, top_k: Optional[int] = None) -> List[int]:
        """
        Classe les pages par pertinence par rapport à la requête
        
        Args:
            pages: Liste des textes de chaque page
            query: Requête utilisateur
            top_k: Nombre de pages à retourner (si None, retourne toutes les pages)
            
        Returns:
            Liste des indices de pages classés par pertinence décroissante
        """
        if not pages:
            return []
        
        # Prétraiter le corpus et calculer les statistiques nécessaires
        tokenized_pages = self.fit(pages)
        
        # Prétraiter la requête
        query_terms = self._preprocess_text(query)
        
        # Étendre la requête avec des termes connexes
        expanded_query_terms = self._expand_query(query_terms)
        
        # Calculer les scores pour chaque page
        scores = []
        for i, page_terms in enumerate(tokenized_pages):
            score = self._calculate_page_score(page_terms, expanded_query_terms, i)
            scores.append(score)
        
        # Trier les indices par score décroissant
        ranked_indices = np.argsort(scores)[::-1].tolist()
        
        # Ajouter une vérification supplémentaire pour les pages avec un score trop faible
        # Pour éviter des correspondances non pertinentes
        threshold = max(scores) * 0.1 if scores else 0
        ranked_indices = [idx for idx in ranked_indices if scores[idx] > threshold]
        
        # Retourner les top_k indices si spécifié
        if top_k is not None and top_k < len(ranked_indices):
            return ranked_indices[:top_k]
        
        return ranked_indices