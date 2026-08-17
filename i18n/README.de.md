[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent ist ein schlankes Tornado-Backend und eine Progressive Web App (PWA) zum Durchsuchen und Vorschau lokaler Organoid-Datensätze mit minimalem Setup. Es bietet eine dateitypspezifische Vorschau für Tabellen, Mikroskopie-Bilder (einschließlich TIFF), Archive, gzip-Text und AnnData `.h5ad`-Analyseobjekte.

## 🎯 Kurzüberblick

| Ziel | Was dieses Repo bietet |
|---|---|
| Lokale, datenzentrierte Erkundung | Datensatzentdeckung, Metadaten und Dateibrowser direkt aus einem lokalen `datasets/`-Arbeitsbereich |
| Umfassende Vorschauen | Vorschaupfade für Tabellen, Bilder (inkl. TIFF), Archive, `.gz` und `.h5ad` |
| Offline-freundliches Frontend | Installierbare PWA-Shell mit Service Worker und Manifest |
| Praktische Abläufe | Entpacken von Archiven + kategoriebasierte Indexierung |

## Überblick 🔭

Die Kernanwendung ist für interaktive Datensatz-Exploration mit minimalem Setup konzipiert:

- Backend-API und Vorschau-Engine in `app.py`
- PWA-Frontend in `web/`
- Download-Helfer in `scripts/`
- Lokaler Datensatz-Arbeitsbereich in `datasets/` (git-ignored)

Dieses Repository enthält außerdem angrenzende Forschungs- und Hilfs-Arbeitsbereiche (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, Untermodul `papers`). Das in dieser README beschriebene Primär-Runtime ist die Top-Level-App `OrganoidAgent`.

## Features ✨

- Lokale Datensatzindizierung mit Größen- und Dateianzahl-Zusammenfassungen
- Rekursive Dateiauflistung mit erkannter Dateikategorie
- Vorschauunterstützung für CSV/TSV/XLS/XLSX-Tabellen
- Vorschauunterstützung für TIFF/JPG/PNG-Bilder
- Vorschauunterstützung für `.h5ad`-Zusammenfassungen mit Embedding/PCA-Scatter-Vorschau
- Vorschauunterstützung für ZIP/TAR/TGZ-Archiv-Listing + erster Bildvorschauversuch
- Vorschau von `.gz`-Text: Ausgabe der ersten Zeilen
- Endpunkt für Archiv-Extraktion bei großen gepackten Datensätzen
- Datensatz-Metadaten-Karten aus Markdown gerendert
- PWA-Frontend mit Service Worker und Manifest
- Einfache Pfadbereinigung (`safe_dataset_path`), um Dateizugriff auf `datasets/` zu beschränken

### Kurzüberblick

| Bereich | Was es bietet |
|---|---|
| Datensatzentdeckung | Auflistung auf Verzeichnisebene mit Dateianzahlen und Größenübersichten |
| Dateiexploration | Rekursive Auflistung und Erkennung von Dateiklassen (`image`, `table`, `analysis`, `archive` usw.) |
| Umfangreiche Vorschauen | Tabellen, TIFF/Bilder, gzip-Textausschnitte, Archivinhalte, AnnData-Zusammenfassungen |
| Analyse-Visualisierungen | `.h5ad`-Scatter-Vorschauen aus `obsm`-Embeddings oder PCA-Fallback |
| Paketunterstützung | Archivlisten + Extraktions-Endpunkt für große komprimierte Bündel |
| Web-UX | Installierbare PWA mit offline-freundlichen Service-Worker-Assets |

## Projektstruktur 🗂️

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
├─ datasets/                      # heruntergeladene Daten und Vorschau-Cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # Submodul: prompt-is-all-you-need
├─ i18n/                          # derzeit für mehrsprachige README-Dateien vorhanden
├─ BioAgent/                      # zugehörig, aber separate App
├─ BioAgentUtils/                 # zugehörige Trainings-/Datenhilfen
├─ references/
├─ results/
└─ vendor/                        # externe Submodule (copilot-sdk, paper-agent, codex)
```

## Voraussetzungen ✅

- Python `3.10+`
- Empfohlene Umgebung: `conda` oder `venv`

Erforderliche/optionale Python-Pakete aus dem Quelltext abgeleitet:

| Paket | Aufgabe |
|---|---|
| `tornado` | Erforderlich für den Serverstart |
| `pandas` | Optional: Unterstützung für Tabellenvorschauen |
| `anndata`, `numpy` | Optional: `.h5ad`-Vorschau und Analyse-Darstellung |
| `Pillow` | Optional: Bildverarbeitung und erzeugte Vorschauen |
| `tifffile` | Optional: TIFF-Vorschau |
| `requests` | Optional: Dataset-Download-Skripte |
| `kaggle` | Optional: Kaggle-Downloads im Skript für Drug Screening |

Hinweis: Für die Top-Level-App existiert derzeit weder eine `requirements.txt`, noch `pyproject.toml` oder `environment.yml`.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (Beispiel)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: nur minimale Laufzeit
pip install tornado
```

## Nutzung 🚀

### Schnellstart

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # optional if you already have the deps
python app.py --port 8080
```

Öffne `http://localhost:8080`.

### API Smoke Test

```bash
curl http://localhost:8080/api/datasets
```

### Datendownload (optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Heruntergeladene Daten liegen in `datasets/` (git-ignored).

## API-Endpunkte 🌐

| Methode | Endpunkt | Zweck |
|---|---|---|
| `GET` | `/api/datasets` | Listet Datensätze mit Statistik-Zusammenfassungen |
| `GET` | `/api/datasets/{name}` | Listet Dateien für einen Datensatz |
| `GET` | `/api/datasets/{name}/metadata` | Liefert Markdown-Metadatenkarte |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Kategorieorientierte Dateiauflistung |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Vorschauantworten mit Dateityp-spezifischer Logik |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Entpackt ein Archiv in einen benachbarten `_extracted`-Ordner |
| `GET` | `/files/<path>` | Direkte Bereitstellung von Rohdateien |
| `GET` | `/previews/<path>` | Bereitstellung generierter Vorschaudateien |

Beispielhafter Vorschauaufruf:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Konfiguration 🧩

Die aktuelle Laufzeitkonfiguration ist bewusst klein gehalten:

- Serverport: `--port`-Argument in `app.py` (Standard `8080`)
- Datensatzverzeichnis: fest auf `datasets/` relativ zum Repository-Root
- Vorschau-Cache: `datasets/.cache/previews`
- Metadaten-Mapping: `DATASET_METADATA`-Dictionary in `app.py`
- GitHub-API-Token für Downloader (optional): Umgebungsvariable `GITHUB_TOKEN` oder `--github-token`

Hinweis: Falls konfigurierbare Datensatz-Wurzeln oder Produktionsserver-Einstellungen nötig sind, sind diese noch nicht in zentralen Konfigurationsdateien verfügbar.

## Beispiele 🧪

### Dateibrowsing nach Kategorie

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Ein Archiv extrahieren

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Selektive Download-Modi ausführen

```bash
# Organoid Datensätze: GEO überspringen, Zenodo behalten
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening Datensätze: nur Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Entwicklungshinweise 🛠️

- Backend liefert statische Frontend-Dateien aus `web/` aus.
- Service Worker und Manifest befinden sich in `web/sw.js` und `web/manifest.json`.
- Dateityp-Routing und Vorschau werden in `app.py` umgesetzt.
- Manuelle Validierung (aktuelle Projektrichtlinie): PWA lädt unter `http://localhost:8080`
- Manuelle Validierung (aktuelle Projektrichtlinie): `/api/datasets` liefert JSON
- Manuelle Validierung (aktuelle Projektrichtlinie): Vorschauen werden für CSV/XLSX/Bilder/Archive gerendert

## Fehlerbehebung 🩺

- `ModuleNotFoundError` für Vorschau-Bibliotheken: Fehlende Pakete installieren (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Leere Datensatzliste: Prüfe, ob Daten unter `datasets/` vorhanden sind und Ordner nicht mit `.` beginnen.
- `.h5ad`-Vorschau zeigt kein Scatterbild: Prüfe, ob `anndata`, `numpy` und `Pillow` installiert sind.
- Probleme bei großer Archivvorschau/Extraktion: Nutze den Extraktions-Endpunkt und prüfe direkt die entpackten Dateien.
- GitHub-Downloader-Rate-Limit-Fehler: `GITHUB_TOKEN` per Umgebungsvariable oder CLI-Flag bereitstellen.
- Kaggle-Download funktioniert nicht: `kaggle` installieren und `~/.kaggle/kaggle.json`-Zugangsdaten konfigurieren.

## Roadmap 🧭

Mögliche nächste Verbesserungen (in der Top-Level-App noch nicht vollständig umgesetzt):

- Wurzel-Dependency-Manifest hinzufügen (`requirements.txt` oder `pyproject.toml`)
- Automatisierte Tests für API-Handler und Vorschaufunktionen ergänzen
- Konfigurierbare Datensatz-Root- und Cache-Einstellungen einführen
- Explizites Produktionslauf-Profil ergänzen (nicht im Debug, Reverse-Proxy-Hinweise)
- Mehrsprachige Dokumentation unter `i18n/` erweitern

## Mitwirken 🤝

Beiträge sind willkommen. Ein praktikabler Workflow:

1. Forken und einen fokussierten Branch erstellen.
2. Änderungen auf einen logischen Bereich beschränken.
3. App-Start und Schlüsselfunktionen manuell validieren.
4. PR mit Zusammenfassung, ausgeführten Befehlen und Screenshots bei UI-Änderungen eröffnen.

Lokale Stilkonventionen in diesem Repository:

- Python: 4-Leerzeichen-Einrückung, snake_case für Funktionen/Dateien, CapWords für Klassen
- Behalte Frontend-Logik in `web/app.js` für diese App (vermeide unnötige Framework-Umbauten)
- Halte Kommentare knapp und nur dort, wo die Logik nicht sofort ersichtlich ist

## Projektstruktur (kanonisch) 📌

- `app.py`: Tornado-Server und API-Routen.
- `web/`: PWA-Assets.
- `scripts/`: Dataset-Download-Helfer.
- `datasets/`: lokaler Datenspeicher.
- `papers/`: Untermodul mit Referenzmaterial.

## Lizenz 📄

Eine top-level `LICENSE`-Datei ist im Repository-Root derzeit nicht vorhanden.

Hinweis: Solange keine Top-Level-Lizenz existiert, bleiben die Bedingungen für Wiederverwendung und Weiterverbreitung für den Top-Level-Code von OrganoidAgent unverändert offen.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
