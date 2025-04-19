from typing import Dict, List, Tuple, Union
import numpy as np

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Implémentation optimisée de la distance de Levenshtein
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        Distance d'édition entre les deux chaînes
    """
    # Cas triviaux
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Créer la matrice (seulement 2 lignes nécessaires)
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)
    
    # Remplir la matrice
    for i in range(1, len(s1) + 1):
        current_row[0] = i
        
        for j in range(1, len(s2) + 1):
            deletion = previous_row[j] + 1
            insertion = current_row[j-1] + 1
            substitution = previous_row[j-1] + (0 if s1[i-1] == s2[j-1] else 1)
            current_row[j] = min(deletion, insertion, substitution)
        
        # Échanger les lignes
        previous_row, current_row = current_row, previous_row
    
    # La réponse est dans la dernière cellule calculée
    return previous_row[len(s2)]

def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Calcule le ratio de similarité basé sur la distance de Levenshtein
    
    Args:
        s1: Première chaîne
        s2: Deuxième chaîne
        
    Returns:
        Ratio de similarité entre 0 et 1 (1 = identique)
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 1.0  # Les deux chaînes vides sont identiques
    
    return 1.0 - (distance / max_len)

def find_closest_text(text: str, candidates: List[str], threshold: float = 0.85) -> Tuple[int, float]:
    """
    Trouve le texte le plus proche parmi une liste de candidats
    
    Args:
        text: Texte de référence
        candidates: Liste de textes candidats
        threshold: Seuil minimal de similarité
        
    Returns:
        Tuple contenant:
        - L'indice du candidat le plus proche (-1 si aucun ne dépasse le seuil)
        - Le score de similarité du meilleur candidat
    """
    if not candidates:
        return -1, 0.0
    
    # Calculer les scores pour tous les candidats
    scores = [levenshtein_ratio(text, candidate) for candidate in candidates]
    max_score_idx = np.argmax(scores)
    max_score = scores[max_score_idx]
    
    # Vérifier le seuil
    if max_score >= threshold:
        return max_score_idx, max_score
    else:
        return -1, max_score

def align_response(response: str, source_sentences: List[str], threshold: float = 0.70) -> List[Dict[str, Union[str, float]]]:
    """
    Aligne la réponse générée avec les phrases sources
    
    Args:
        response: Réponse générée par l'IA
        source_sentences: Phrases sources du document
        threshold: Seuil minimal de similarité
        
    Returns:
        Liste de dictionnaires contenant:
        - 'generated': Phrase générée
        - 'source': Phrase source correspondante (ou None)
        - 'similarity': Score de similarité
        - 'aligned': Booléen indiquant si la phrase est alignée
    """
    if not response or not source_sentences:
        return []
    
    # Nettoyer la réponse pour s'assurer qu'elle est correctement formatée
    response = clean_text(response)
    
    # Découper la réponse en phrases de manière plus robuste en utilisant NLTK
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download('punkt', quiet=True)
        response_sentences = sent_tokenize(response)
    except:
        # Fallback en cas d'erreur avec NLTK
        response_sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
    
    # Filtrer les phrases trop courtes
    response_sentences = [s.strip() for s in response_sentences if len(s.strip()) > 10]
    
    # Pour chaque phrase de la réponse, trouver la phrase source la plus proche
    results = []
    for resp_sent in response_sentences:
        idx, score = find_closest_text(resp_sent, source_sentences, threshold)
        
        if idx >= 0:
            results.append({
                'generated': resp_sent,
                'source': source_sentences[idx],
                'similarity': score,
                'aligned': True
            })
        else:
            # Si aucune correspondance n'est trouvée, chercher la meilleure correspondance
            # même si elle est sous le seuil
            best_idx, best_score = -1, 0
            for i, source in enumerate(source_sentences):
                current_score = levenshtein_ratio(resp_sent, source)
                if current_score > best_score:
                    best_score = current_score
                    best_idx = i
            
            if best_idx >= 0 and best_score >= 0.4:  # Utiliser un seuil réduit pour les cas limites
                results.append({
                    'generated': resp_sent,
                    'source': source_sentences[best_idx],
                    'similarity': best_score,
                    'aligned': False  # Toujours marqué comme non aligné pour la transparence
                })
            else:
                results.append({
                    'generated': resp_sent,
                    'source': None,
                    'similarity': best_score if best_idx >= 0 else 0,
                    'aligned': False
                })
    
    return results

def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les caractères d'échappement et en normalisant les espaces.
    Cette fonction est utilisée pour nettoyer les textes dans la sortie JSON, pas pour les textes extraits de PDF.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    import re
    
    # Remplacer les caractères d'échappement courants
    replacements = {
        '\\n': ' ', # Saut de ligne
        '\\t': ' ', # Tabulation
        '\\r': '',  # Retour chariot
        '\\\"': '"', # Guillemets échappés
        '\\\'': "'", # Apostrophe échappée
        '\\\\': '\\', # Backslash échappé
        '\\u00e9': 'é', # é
        '\\u00e8': 'è', # è
        '\\u00ea': 'ê', # ê
        '\\u00e0': 'à', # à
        '\\u00e2': 'â', # â
        '\\u00e7': 'ç', # ç
        '\\u00f4': 'ô', # ô
        '\\u00fb': 'û', # û
        '\\u00ee': 'î', # î
        '\\u00ef': 'ï', # ï
        '\\u00fc': 'ü', # ü
        '\\u0153': 'œ', # œ
        '\\u2019': "'", # apostrophe typographique
        '\\u2026': '...', # points de suspension
    }
    
    # Appliquer les remplacements
    for esc, char in replacements.items():
        text = text.replace(esc, char)
    
    # Supprimer les caractères Unicode échappés restants
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Normaliser la ponctuation: supprimer les espaces avant la ponctuation
    text = re.sub(r'\s+([,.;:!?)])', r'\1', text)
    # Normaliser la ponctuation: ajouter un espace après la ponctuation si ce n'est pas déjà le cas
    text = re.sub(r'([,.;:!?)])(?!\s|$)', r'\1 ', text)
    
    return text.strip()

def build_factual_response(alignment_results: List[Dict[str, Union[str, float]]]) -> str:
    """
    Construit une réponse factuelle à partir des résultats d'alignement
    
    Args:
        alignment_results: Résultats de la fonction align_response
        
    Returns:
        Réponse factuelle basée uniquement sur les phrases sources
    """
    # Si au moins une phrase a été considérée comme alignée, l'utiliser
    aligned_parts = [result['source'] for result in alignment_results if result['aligned'] and result['source']]
    
    # Si aucune phrase alignée, mais des correspondances ont été trouvées avec un bon score
    if not aligned_parts:
        # Extraire les meilleurs correspondances, même si elles sont sous le seuil d'alignement
        best_matches = []
        for result in alignment_results:
            # Si nous avons une phrase source et un score de similarité >= 0.3
            if result['source'] and result['similarity'] >= 0.3:
                best_matches.append((result['source'], result['similarity']))
        
        # Trier par score de similarité décroissant et prendre les 5 meilleures correspondances
        best_matches.sort(key=lambda x: x[1], reverse=True)
        aligned_parts = [match[0] for match in best_matches[:5]]
    
    if not aligned_parts:
        # Si aucune correspondance satisfaisante n'a été trouvée, utiliser les phrases brutes sources
        return "Impossible de répondre à cette question en se basant uniquement sur le document fourni."
    
    # Construire la réponse à partir des phrases sources
    return ". ".join(aligned_parts) + "."