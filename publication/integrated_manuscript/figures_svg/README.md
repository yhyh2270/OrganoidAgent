# Editable legacy figure assets

The nine legacy PDF files were inspected before conversion. Each PDF contains a
single full-page JPEG, so it has no recoverable vector text, paths, arrows, or
chart data.

This directory therefore uses a two-level migration strategy:

- `legacy_canvases/`: SVG editing canvases with the original flattened figure
  isolated in the `legacy-raster-reference` layer. These preserve the exact old
  appearance and provide a stable canvas for panel-by-panel replacement.
- `editable_components/`: genuinely editable SVG reconstructions. Text, boxes,
  arrows, connectors, colors, and node positions are native SVG objects.
- `assets/`: high-resolution legacy PNG references used by the canvases.
- `conversion_manifest.json`: source provenance, dimensions, and output mapping.

## Current native-vector components

- `figure1_data_pipeline.svg`: reusable data-to-analysis workflow for the new
  integrated workstation figure.
- `figure2_segmentation_pipeline.svg`: multiscale Cellpose and recovery workflow
  reconstructed from the old Figure 2B.
- `figure2_aaps_fully_editable.svg`: full native-vector reconstruction of the
  legacy five-panel AAPS figure. Panels, labels, boxes, connectors, ports, and
  icons are separate SVG objects organized by descriptive IDs.
- `figure2_integrated_organoid_workflow.svg`: publication-oriented revision of
  the AAPS figure with parallel Morphological Analysis and Viability Detection
  branches, explicit execution policy, and a unified evidence package.

Microscopy panels should remain raster images in the final SVG. Statistical
plots should be regenerated from their CSV source as SVG/PDF rather than traced
from pixels. This preserves quantitative accuracy and editable axes, legends,
and curves.

Open the files in Inkscape, Illustrator, Affinity Designer, or recent
PowerPoint versions. Keep the legacy reference layer locked while rebuilding a
panel, then hide it before export.

Run `python build_legacy_svg_assets.py` to rebuild the canvases and components.
