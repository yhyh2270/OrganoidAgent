[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent est un backend Tornado léger et une Progressive Web App (PWA) pour parcourir et prévisualiser localement des jeux de données d’organoïdes avec une configuration minimale. Il propose un rendu de prévisualisation selon le type de fichier pour les tableaux, les images de microscopie (y compris TIFF), les archives, les fichiers texte gzip et les objets d’analyse AnnData `.h5ad`.

## 🎯 En bref

| Objectif | Ce que ce dépôt vous apporte |
|---|---|
| Exploration locale d’abord | Découverte de jeux de données, métadonnées et navigation des fichiers depuis un espace de travail local `datasets/` |
| Prévisualisations riches | Parcours de tableaux, images, archives, `.gz` et `.h5ad` via des chemins de prévisualisation dédiés |
| Frontend orienté hors ligne | Shell PWA installable avec service worker et manifeste |
| Opérations pratiques | Extraction d’archives + indexation par filtres de catégorie |

## Vue d’ensemble 🔭

L’application principale est conçue pour l’exploration interactive de jeux de données avec une configuration minimale :

- API backend et moteur de prévisualisation dans `app.py`
- Frontend PWA dans `web/`
- Helpers de téléchargement dans `scripts/`
- Espace de travail local des jeux de données dans `datasets/` (ignoré par git)

Ce dépôt contient également des espaces de travail de recherche et d’utilitaires associés (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, sous-module `papers`). L’exécution principale décrite dans ce README correspond à l’application `OrganoidAgent` au niveau racine.

## Fonctionnalités ✨

- Indexation locale des jeux de données avec résumés de taille et de nombre de fichiers
- Liste récursive des fichiers de jeu de données avec détection du type de fichier
- Prévisualisation des tableaux CSV/TSV/XLS/XLSX
- Prévisualisation d’images TIFF/JPG/PNG
- Prévisualisation des résumés `.h5ad` avec génération d’un scatter plot d’embeddings/PCA
- Prévisualisation des archives ZIP/TAR/TGZ avec tentative d’affichage de la première image
- Prévisualisation des premières lignes de texte `.gz`
- Endpoint d’extraction d’archives pour les grands jeux de données empaquetés
- Cartes de métadonnées au niveau du jeu de données rendues depuis Markdown
- Frontend PWA avec service worker et manifeste
- Assainissement de chemin de base (`safe_dataset_path`) pour restreindre l’accès aux fichiers sous `datasets/`

### En un coup d’œil

| Domaine | Ce qu’il fournit |
|---|---|
| Découverte de jeu de données | Liste des jeux de données au niveau dossier avec nombre de fichiers et résumés de taille |
| Exploration de fichiers | Liste récursive et détection du type (`image`, `table`, `analysis`, `archive`, etc.) |
| Prévisualisations riches | Tableaux, images TIFF, extraits texte gzip, contenu d’archives, résumés AnnData |
| Visualisations d’analyse | Prévisualisations de scatter `.h5ad` depuis les embeddings `obsm` ou fallback PCA |
| Support de packaging | Listing d’archives + endpoint d’extraction pour les bundles compressés volumineux |
| Expérience Web | PWA installable avec assets de service worker adaptés au hors-ligne |

## Structure du projet 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Prérequis ✅

- Python `3.10+`
- Gestionnaire d’environnement recommandé : `conda` ou `venv`

Packages Python requis/optionnels déduits du code source :

| Package | Rôle |
|---|---|
| `tornado` | Requis pour le démarrage du serveur |
| `pandas` | Optionnel : support de prévisualisation des tableaux |
| `anndata`, `numpy` | Optionnel : prévisualisation `.h5ad` et tracés d’analyse |
| `Pillow` | Optionnel : rendu d’image et génération de prévisualisations |
| `tifffile` | Optionnel : support de prévisualisation TIFF |
| `requests` | Optionnel : scripts de téléchargement de jeux de données |
| `kaggle` | Optionnel : téléchargements Kaggle dans le script de drug screening |

Remarque sur les hypothèses : aucun `requirements.txt`, `pyproject.toml` ou `environment.yml` n’existe actuellement pour l’app de niveau racine.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (example)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: minimal runtime only
pip install tornado
```

## Utilisation 🚀

### Démarrage rapide

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Ouvrez `http://localhost:8080`.

### Test rapide de l’API

```bash
curl http://localhost:8080/api/datasets
```

### Télécharger les données (optionnel)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Les données téléchargées sont stockées dans `datasets/` (ignoré par git).

## Endpoints API 🌐

| Méthode | Endpoint | Objectif |
|---|---|---|
| `GET` | `/api/datasets` | Lister les jeux de données avec statistiques de résumé |
| `GET` | `/api/datasets/{name}` | Lister les fichiers d’un jeu de données |
| `GET` | `/api/datasets/{name}/metadata` | Retourner la carte de métadonnées Markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Liste de fichiers orientée catégorie |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Charge utile de prévisualisation adaptée au type de fichier |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Extraire l’archive dans un dossier `_extracted` frère |
| `GET` | `/files/<path>` | Servir le fichier dataset brut |
| `GET` | `/previews/<path>` | Servir les assets de prévisualisation générés |

Exemple d’appel de prévisualisation :

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuration 🧩

La configuration runtime actuelle est volontairement réduite :

- Port serveur : argument `--port` dans `app.py` (valeur par défaut `8080`)
- Répertoire de données : fixe sur `datasets/` depuis la racine du dépôt
- Cache de prévisualisation : `datasets/.cache/previews`
- Mapping des métadonnées : dictionnaire `DATASET_METADATA` dans `app.py`
- Jeton GitHub API pour le downloader (optionnel) : variable d’environnement `GITHUB_TOKEN` ou `--github-token`

Remarque sur les hypothèses : si vous avez besoin de racines de jeux de données configurables ou de paramètres serveur de production, ils ne sont pas encore exposés dans des fichiers de configuration de haut niveau.

## Exemples 🧪

### Parcourir des fichiers par catégorie

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extraire une archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Exécuter des modes de téléchargement sélectifs

```bash
# Jeux de données organoïdes : ignorer GEO, conserver Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Jeux de données de criblage pharmacologique : seulement Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Notes de développement 🛠️

- Le backend sert les assets frontend statiques depuis `web/`.
- Le service worker et le manifeste se trouvent dans `web/sw.js` et `web/manifest.json`.
- Le routage selon le type de fichier et les prévisualisations sont implémentés dans `app.py`.
- Validation manuelle (guide actuelle du projet) : la PWA se charge sur `http://localhost:8080`
- Validation manuelle (guide actuelle du projet) : `/api/datasets` renvoie du JSON
- Validation manuelle (guide actuelle du projet) : les prévisualisations s’affichent pour CSV/XLSX/images/archives

## Dépannage 🩺

- `ModuleNotFoundError` pour les bibliothèques de prévisualisation : installez les paquets manquants (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Liste vide de jeux de données : vérifiez que des données existent dans `datasets/` et que les dossiers ne commencent pas par un point.
- Prévisualisation `.h5ad` sans scatter image : vérifiez que `anndata`, `numpy` et `Pillow` sont installés.
- Problèmes de prévisualisation/extraction de grosses archives : utilisez l’endpoint d’extraction et inspectez directement les fichiers extraits.
- Erreurs de quota de taux du downloader GitHub : fournissez `GITHUB_TOKEN` via la variable d’environnement ou le flag CLI.
- Téléchargement Kaggle non fonctionnel : installez `kaggle` et configurez les identifiants dans `~/.kaggle/kaggle.json`.

## Feuille de route 🧭

Améliorations potentielles (pas encore totalement implémentées dans cette app racine) :

- Ajouter un manifeste de dépendances racine (`requirements.txt` ou `pyproject.toml`)
- Ajouter des tests automatisés pour les handlers API et les fonctions de prévisualisation
- Ajouter des réglages configurables pour la racine des données et du cache
- Ajouter un profil d’exécution explicitement orienté production (non debug, guidance reverse-proxy)
- Étendre la documentation multilingue sous `i18n/`

## Contribution 🤝

Les contributions sont bienvenues. Un flux de travail pratique :

1. Forkez et créez une branche ciblée.
2. Gardez les changements centrés sur une zone logique unique.
3. Validez manuellement le démarrage de l’application et les endpoints clés.
4. Ouvrez une PR avec un résumé, les commandes exécutées, et des captures d’écran pour les changements UI.

Conventions de style locales dans ce dépôt :

- Python : indentation de 4 espaces, fonctions/fichiers en snake_case, classes en CapWords
- Conserver la logique frontend dans `web/app.js` pour cette app (éviter les réécritures de framework inutiles)
- Garder les commentaires concis et seulement quand la logique n’est pas évidente

## Synthèse du projet (référence) 📌

- `app.py` : serveur Tornado et routes API.
- `web/` : assets PWA.
- `scripts/` : helpers de téléchargement de jeux de données.
- `datasets/` : stockage local des données.
- `papers/` : sous-module contenant des documents de référence.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

Aucun fichier `LICENSE` de projet n’est actuellement présent à la racine de ce dépôt.

Remarque sur les hypothèses : tant qu’aucune licence racine n’est ajoutée, les conditions de réutilisation/redistribution restent non précisées pour la base de code OrganoidAgent de niveau racine.
