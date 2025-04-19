# Veritas

Veritas est une chaîne d'agents IA qui répond à des questions sur des documents PDF en utilisant **uniquement** des phrases extraites du document, garantissant des réponses factuelles et sans hallucination.

## Principe

La chaîne de traitement de Veritas fonctionne comme suit :

1. 📥 **Input** : Un fichier PDF et une question
2. 🔍 **Indexation améliorée** : Expansion de requête pour transformer la question en pseudo-réponse puis recherche BM25
3. 🧑‍⚖️ **Agent 1** : Sélection des pages réellement pertinentes parmi celles présélectionnées
4. ✂️ **Découpage** : Segmentation du texte en phrases individuelles (granularité fine pour un meilleur contrôle)
5. 🚫 **Agent 2** : Élimination des phrases non pertinentes pour constituer un "sac de phrases"
6. 🗣️ **Agent 3** : Génération d'une réponse basée **uniquement** sur les phrases sélectionnées
7. 📐 **Alignement Levenshtein** : Vérification que chaque partie de la réponse est bien extraite du document original

## Nouveautés

- **Outil de nettoyage de PDF** : Prétraitement intelligent des PDFs avec formatage IA
- **Expansion de requête** : Transformation des questions en requêtes enrichies pour de meilleurs résultats
- **Respect strict du texte original** : Garantie que l'agent 3 utilise les phrases exactes sans modification
- **Formats multiples** : Export en PDF, texte ou JSON

## Installation

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'API key (copier et modifier le fichier d'exemple)
cp config/.env.example config/.env
# Éditer config/.env pour ajouter votre API key
```

## Utilisation

### 1. Nettoyage des PDFs (recommandé)

Avant d'interroger un document, il est recommandé de le nettoyer pour améliorer la qualité des réponses :

```bash
# Générer un nouveau PDF avec texte nettoyé (par défaut)
./cleanPdf.py document.pdf

# Autres formats disponibles
./cleanPdf.py document.pdf --format text  # Fichier texte
./cleanPdf.py document.pdf --format json  # Extraction de phrases en JSON

# Options
./cleanPdf.py document.pdf --output chemin/sortie.pdf --verbose
```

### 2. Interrogation du document

Une fois le document nettoyé, vous pouvez l'interroger avec des questions en langage naturel :

```bash
# Interroger le document nettoyé
./veritas.py document_clean.pdf "Quelle est la définition de X dans ce document?"

# Ou interroger directement le PDF original (moins précis)
./veritas.py document.pdf "Votre question sur le document?"

# Options avancées
./veritas.py document.pdf "Ma question?" --verbose --debug --output rapport.json

# Désactiver l'expansion de requête pour BM25
./veritas.py document.pdf "Ma question?" --no-query-expansion
```

## Options

### Options de veritas.py

- `--output`, `-o` : Chemin vers un fichier de sortie pour enregistrer les détails complets en JSON
- `--verbose`, `-v` : Afficher des informations détaillées sur le processus
- `--debug`, `-d` : Activer le mode debug avec plus d'informations (requêtes enrichies, etc.)
- `--no-query-expansion` : Désactiver l'expansion de requête pour BM25

### Options de cleanPdf.py

- `--output`, `-o` : Chemin de sortie personnalisé pour le fichier généré
- `--format`, `-f` : Format de sortie (`pdf`, `text` ou `json`)
- `--verbose`, `-v` : Afficher des informations détaillées sur le processus

## Exemples

```bash
# Nettoyer un document sur le RGPD
./cleanPdf.py rgpd.pdf

# Poser une question sur les droits des personnes concernées
./veritas.py rgpd_clean.pdf "Quels sont les droits des personnes concernées selon le RGPD?"

# Question sur les obligations des responsables de traitement avec debug
./veritas.py rgpd_clean.pdf "Quelles sont les obligations des responsables de traitement?" --debug
```

## Configuration

Les paramètres configurables sont disponibles dans le fichier `config/.env` :

- `CREW_API_KEY` : Clé API pour les modèles de langage
- `CREW_BASE_URL` : URL de base pour l'API (par défaut : "https://openrouter.ai/api/v1")
- `CREW_MODEL` : Modèle à utiliser (par défaut : "openrouter/openai/gpt-4.1-mini")
- `CREW_TEMPERATURE` : Température pour la génération (par défaut : 0.7)
- `CREW_MAX_TOKENS` : Nombre maximum de tokens (par défaut : 4000)
- `MIN_SIMILARITY_THRESHOLD` : Seuil de similarité Levenshtein (par défaut : 0.75)
- `BM25_TOP_K` : Nombre de pages à présélectionner par BM25 (par défaut : 20)
- `DEBUG` : Mode debug (par défaut : False)
- `QUERY_EXPANSION` : Activation de l'expansion de requête (par défaut : True)

## Dépendances principales

- crewai : Orchestration des agents IA
- pypdf : Extraction de texte depuis des PDFs
- reportlab : Génération de PDF
- scikit-learn : Implémentation de l'algorithme BM25
- python-Levenshtein : Calcul des distances d'édition
- nltk : Découpage en phrases
- ftfy & unidecode : Nettoyage et normalisation de texte

## Architecture du projet

```
veritas/
├── config/
│   └── .env.example       # Configuration par défaut
├── lib/
│   ├── __init__.py
│   ├── agents.py          # Définition des agents IA
│   ├── bm25.py            # Implémentation de l'algorithme BM25
│   ├── config.py          # Gestion de la configuration
│   ├── levenshtein.py     # Fonctions d'alignement de texte
│   └── pdf_parser.py      # Extraction et découpage du texte PDF
├── cleanPdf.py            # Outil de nettoyage de PDF
├── veritas.py             # Point d'entrée principal
└── requirements.txt       # Dépendances
```

## Limitations

- Performance optimale sur des documents bien structurés et formatés
- Nécessite un document PDF de qualité pour une extraction efficace
- Les réponses sont limitées au contenu explicite du document
- Le nettoyage de grands documents peut prendre du temps