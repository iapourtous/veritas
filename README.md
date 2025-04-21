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


## Installation

### Installation à partir des sources

```bash
# Cloner le dépôt
git clone https://github.com/iapourtous/veritas.git
cd veritas

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer le package en mode développement
pip install -e .

# Configurer l'API key (copier et modifier le fichier d'exemple)
cp config/.env.example config/.env
# Éditer config/.env pour ajouter votre API key
```

### Installation via pip (à venir)

```bash
pip install veritas
```

## Utilisation

### En mode développement (après pip install -e .)

Après l'installation en mode développement, vous pouvez utiliser Veritas de deux façons :

```bash
# Utilisation directe avec python -m
python -m veritas.cli document.pdf "Ma question?"
python -m veritas.clean_pdf document.pdf --format text

# OU en utilisant les points d'entrée installés (selon votre environnement)
veritas document.pdf "Ma question?"
cleanpdf document.pdf
```

### 1. Nettoyage des PDFs (recommandé)

Avant d'interroger un document, il est recommandé de le nettoyer pour améliorer la qualité des réponses :

```bash
# Générer un nouveau PDF avec texte nettoyé (par défaut)
python -m veritas.clean_pdf document.pdf

# Autres formats disponibles
python -m veritas.clean_pdf document.pdf --format text  # Fichier texte
python -m veritas.clean_pdf document.pdf --format json  # Extraction de phrases en JSON

# Options
python -m veritas.clean_pdf document.pdf --output chemin/sortie.pdf --verbose
```

### 2. Interrogation du document

Une fois le document nettoyé, vous pouvez l'interroger avec des questions en langage naturel :

```bash
# Interroger le document nettoyé
python -m veritas.cli document_clean.pdf "Quelle est la définition de X dans ce document?"

# Ou interroger directement le PDF original (moins précis)
python -m veritas.cli document.pdf "Votre question sur le document?"

# Options avancées
python -m veritas.cli document.pdf "Ma question?" --verbose --debug --output rapport.json

# Désactiver l'expansion de requête pour BM25
python -m veritas.cli document.pdf "Ma question?" --no-query-expansion
```

## Options

### Options de veritas.cli

- `--output`, `-o` : Chemin vers un fichier de sortie pour enregistrer les détails complets en JSON
- `--verbose`, `-v` : Afficher des informations détaillées sur le processus
- `--debug`, `-d` : Activer le mode debug avec plus d'informations (requêtes enrichies, etc.)
- `--no-query-expansion` : Désactiver l'expansion de requête pour BM25

### Options de veritas.clean_pdf

- `--output`, `-o` : Chemin de sortie personnalisé pour le fichier généré
- `--format`, `-f` : Format de sortie (`pdf`, `text` ou `json`)
- `--verbose`, `-v` : Afficher des informations détaillées sur le processus

## Exemples

```bash
# Exemple concret avec le fichier alice.pdf fourni dans le projet
python -m veritas.cli alice.pdf "Comment s'appelle le chat d'Alice ?"
# Résultat :
# ================================================================================
# QUESTION: Comment s'appelle le chat d'Alice ?
# --------------------------------------------------------------------------------
# RÉPONSE: Pourtant je voudrais bien vous montrer Dinah, notre chatte..
# --------------------------------------------------------------------------------
# SOURCES:
# - Page 28
# ================================================================================

```

## Configuration

La configuration de Veritas utilise une approche hybride combinant fichiers YAML et variables d'environnement:

1. **Fichiers YAML** : Contiennent la configuration de base dans `config/yaml/`
   - `defaults.yaml` : Configuration par défaut (modèles, paramètres, etc.)
   - `agents.yaml` : Configuration des agents (rôles, objectifs, etc.)
   - `prompts.yaml` : Templates de prompts pour les agents

2. **Variables d'environnement** : Définies dans `config/.env` pour remplacer les valeurs par défaut
   - `CREW_API_KEY` : Clé API pour les modèles de langage
   - `CREW_BASE_URL` : URL de base pour l'API
   - `CREW_MODEL` : Modèle à utiliser
   - `CREW_TEMPERATURE` : Température pour la génération
   - `CREW_MAX_TOKENS` : Nombre maximum de tokens
   - `MIN_SIMILARITY_THRESHOLD` : Seuil de similarité Levenshtein
   - `BM25_TOP_K` : Nombre de pages à présélectionner par BM25
   - `DEBUG` : Mode debug
   - `QUERY_EXPANSION` : Activation de l'expansion de requête

### Priorité des configurations

1. Variables d'environnement (priorité la plus haute)
2. Fichier .env
3. Fichiers YAML (priorité la plus basse)

## Dépendances principales

- crewai : Orchestration des agents IA
- pypdf : Extraction de texte depuis des PDFs
- reportlab : Génération de PDF
- scikit-learn : Implémentation de l'algorithme BM25
- python-Levenshtein : Calcul des distances d'édition
- nltk : Découpage en phrases
- ftfy & unidecode : Nettoyage et normalisation de texte
- pyyaml : Gestion des fichiers de configuration YAML

## Architecture du projet

```
veritas/
├── config/
│   ├── .env.example        # Exemple de variables d'environnement
│   └── yaml/               # Configuration en YAML
│       ├── defaults.yaml   # Configuration par défaut
│       ├── agents.yaml     # Configuration des agents
│       └── prompts.yaml    # Templates de prompts
├── src/
│   ├── veritas/            # Package principal
│   │   ├── __init__.py     # Initialisation du package
│   │   ├── clean_pdf.py    # Module de nettoyage de PDF
│   │   ├── cli.py          # Interface en ligne de commande
│   │   └── core.py         # Fonctionnalités principales
│   └── lib/                # Bibliothèques partagées
│       ├── __init__.py
│       ├── agents.py       # Définition des agents IA
│       ├── bm25.py         # Implémentation de l'algorithme BM25
│       ├── config.py       # Interface de configuration
│       ├── yaml_config.py  # Gestionnaire de configuration YAML
│       ├── levenshtein.py  # Fonctions d'alignement de texte
│       └── pdf_parser.py   # Extraction et découpage du texte PDF
├── pyproject.toml          # Configuration du package
├── README.md               # Documentation
└── requirements.txt        # Dépendances (pour rétrocompatibilité)
```

## Limitations

- Performance optimale sur des documents bien structurés et formatés
- Nécessite un document PDF de qualité pour une extraction efficace
- Les réponses sont limitées au contenu explicite du document
- Le nettoyage de grands documents peut prendre du temps