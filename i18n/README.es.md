[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent es un backend liviano con Tornado y una Progressive Web App (PWA) para explorar y previsualizar datasets de organoides localmente con mínima configuración. Ofrece renderizado de previsualización sensible al tipo de archivo para tablas, imágenes de microscopía (incluido TIFF), archivos comprimidos, texto gzip y objetos de análisis AnnData `.h5ad`.

## 🎯 En un vistazo

| Objetivo | Lo que aporta este repositorio |
|---|---|
| Exploración de datasets local-first | Descubrimiento de datasets, metadatos y navegación de archivos desde un workspace local `datasets/` |
| Previsualizaciones ricas | Rutas de previsualización para tablas, imágenes (incluido TIFF), archivos comprimidos, `.gz` y `.h5ad` |
| Frontend apto para uso offline | Shell PWA instalable con service worker y manifest |
| Operación práctica | Extracción de archivos y rutas de indexado filtradas por categoría |

## Overview 🔭

La app principal está pensada para la exploración interactiva de datasets con mínima configuración:

- API backend y motor de previsualización en `app.py`
- Frontend PWA en `web/`
- Utilidades de descarga en `scripts/`
- Workspace local de datasets en `datasets/` (excluido por git)

Este repositorio también contiene workspaces de investigación y utilidades relacionados (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, submódulo `papers`). El runtime principal descrito en este README es la aplicación `OrganoidAgent` de nivel superior.

## Features ✨

- Indexación local de datasets con resúmenes de tamaño y cantidad de archivos
- Listado recursivo de archivos del dataset con inferencia del tipo de archivo
- Soporte de previsualización para tablas CSV/TSV/XLS/XLSX
- Soporte de previsualización para imágenes TIFF/JPG/PNG
- Soporte de previsualización para resúmenes de `.h5ad` con generación de vista previa de scatter de embeddings/PCA
- Soporte de previsualización para listados ZIP/TAR/TGZ + intento de previsualización de la primera imagen
- Soporte de previsualización para primeras líneas de texto `.gz`
- Endpoint de extracción de archivos comprimidos para datasets grandes
- Tarjetas de metadatos a nivel de dataset renderizadas desde Markdown
- Frontend PWA con service worker y manifest
- Sanitización básica de rutas (`safe_dataset_path`) para limitar el acceso de archivos a `datasets/`

### At a glance

| Área | Qué aporta |
|---|---|
| Descubrimiento de datasets | Listado de datasets a nivel de directorio con recuento de archivos y resúmenes de tamaño |
| Exploración de archivos | Listado recursivo e inferencia de tipo (`image`, `table`, `analysis`, `archive`, etc.) |
| Previsualizaciones ricas | Tablas, TIFF/imágenes, fragmentos de texto gzip, contenido de archivos comprimidos, resúmenes de AnnData |
| Visualizaciones de análisis | Previews de dispersión `.h5ad` desde embeddings `obsm` o fallback a PCA |
| Soporte de empaquetado | Listado de archivos comprimidos + endpoint de extracción para bundles grandes |
| UX web | PWA instalable con assets de service worker amigables para uso offline |

## Project Structure 🗂️

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

## Prerequisites ✅

- Python `3.10+`
- Gestor de entorno recomendado: `conda` o `venv`

Paquetes de Python requeridos/opcionales inferidos del código fuente:

| Paquete | Rol |
|---|---|
| `tornado` | Requerido para iniciar el servidor |
| `pandas` | Opcional: soporte de previsualización de tablas |
| `anndata`, `numpy` | Opcional: previsualización `.h5ad` y trazado de análisis |
| `Pillow` | Opcional: renderizado de imágenes y previsualizaciones generadas |
| `tifffile` | Opcional: soporte de previsualización TIFF |
| `requests` | Opcional: scripts de descarga de datasets |
| `kaggle` | Opcional: descargas de Kaggle en el script de screening de fármacos |

Nota de suposición: actualmente no existe `requirements.txt`, `pyproject.toml` ni `environment.yml` para la app de nivel superior.

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Opción A: conda (ejemplo)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Opción B: solo runtime mínimo
pip install tornado
```

## Uso 🚀

### Inicio rápido

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # opcional si ya tienes las dependencias
python app.py --port 8080
```

Abre `http://localhost:8080`.

### Prueba rápida de API

```bash
curl http://localhost:8080/api/datasets
```

### Descargar datos (opcional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Los datos descargados viven en `datasets/` (excluidos de git).

## API Endpoints 🌐

| Método | Endpoint | Propósito |
|---|---|---|
| `GET` | `/api/datasets` | Listado de datasets con estadísticas resumen |
| `GET` | `/api/datasets/{name}` | Listado de archivos de un dataset |
| `GET` | `/api/datasets/{name}/metadata` | Devuelve la tarjeta de metadatos en markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Listado de archivos orientado por categoría |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Payload de previsualización sensible al tipo de archivo |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Extrae un archivo comprimido en una carpeta hermana `_extracted` |
| `GET` | `/files/<path>` | Servicio de archivo de dataset sin procesar |
| `GET` | `/previews/<path>` | Servicio de assets de previsualización generados |

Ejemplo de llamada de previsualización:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuración 🧩

La configuración de runtime actual es intencionalmente pequeña:

- Puerto del servidor: argumento `--port` en `app.py` (por defecto `8080`)
- Directorio de datos: fijo en `datasets/` relativo a la raíz del repositorio
- Caché de previsualizaciones: `datasets/.cache/previews`
- Mapeo de metadatos: diccionario `DATASET_METADATA` en `app.py`
- Token de GitHub para el descargador (opcional): variable de entorno `GITHUB_TOKEN` o flag `--github-token`

Nota de suposición: si necesitas raíces de dataset configurables o ajustes de servidor de producción, aún no están expuestos en archivos de configuración de alto nivel.

## Examples 🧪

### Explorar archivos por categoría

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extraer un archivo comprimido

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Ejecutar modos de descarga selectivos

```bash
# Datasets de organoides: omitir GEO, mantener Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Datasets de screening de fármacos: solo Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Development Notes 🛠️

- El backend sirve los assets estáticos del frontend desde `web/`.
- El service worker y el manifest están en `web/sw.js` y `web/manifest.json`.
- El enrutamiento por tipo de archivo y las previsualizaciones están implementados en `app.py`.
- Validación manual (guía del proyecto actual): la PWA carga en `http://localhost:8080`
- Validación manual (guía del proyecto actual): `/api/datasets` devuelve JSON
- Validación manual (guía del proyecto actual): las previsualizaciones se renderizan para CSV/XLSX/imágenes/archivos comprimidos

## Troubleshooting 🩺

- `ModuleNotFoundError` para librerías de previsualización: instala los paquetes faltantes (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Listado de dataset vacío: confirma que los datos existen en `datasets/` y que los directorios no empiezan por punto.
- Falta la vista previa de `.h5ad`: verifica que `anndata`, `numpy` y `Pillow` estén instalados.
- Problemas con previsualización/extracción de archivos comprimidos grandes: usa el endpoint de extracción e inspecciona directamente los archivos extraídos.
- Errores de cuota por limitación de tasa en el descargador de GitHub: proporciona `GITHUB_TOKEN` mediante variable de entorno o flag CLI.
- Descarga de Kaggle no funciona: instala `kaggle` y configura las credenciales en `~/.kaggle/kaggle.json`.

## Roadmap 🧭

Posibles próximas mejoras (aún no implementadas completamente en esta app de raíz):

- Añadir manifiesto de dependencias en raíz (`requirements.txt` o `pyproject.toml`)
- Añadir pruebas automatizadas para handlers de API y funciones de previsualización
- Añadir configuración configurable para la raíz de datasets y caché
- Añadir perfil de ejecución explícito para producción (no debug, guía de reverse-proxy)
- Expandir documentación multilingüe en `i18n/`

## Contributing 🤝

Las contribuciones son bienvenidas. Un flujo práctico:

1. Haz fork y crea una rama enfocada.
2. Mantén los cambios acotados a una sola área lógica.
3. Valida manualmente el arranque de la app y los endpoints clave.
4. Abre un PR con resumen, comandos ejecutados y capturas de pantalla para cambios de UI.

Convenciones de estilo locales en este repositorio:

- Python: indentación de 4 espacios, funciones/archivos en snake_case, clases en CapWords
- Mantén la lógica frontend en `web/app.js` para esta app (evita reescrituras innecesarias de framework)
- Mantén comentarios concisos y solo donde la lógica no sea obvia

## Project Layout (Canonical Summary) 📌

- `app.py`: servidor Tornado y rutas de API.
- `web/`: assets de la PWA.
- `scripts/`: utilidades de descarga de datasets.
- `datasets/`: almacenamiento local de datos.
- `papers/`: submódulo con materiales de referencia.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

No existe actualmente un archivo `LICENSE` de proyecto en la raíz de este repositorio.

Nota de suposición: hasta que se agregue una licencia de raíz, trata los términos de reutilización/restricción como no especificados para el código base de OrganoidAgent de nivel superior.
