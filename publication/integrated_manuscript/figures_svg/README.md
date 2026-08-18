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
- `figure2_integrated_workflow_4panel.svg`: recommended main-text version. It
  uses a four-part scientific narrative—data context, constrained orchestration,
  parallel morphology/viability workflows, and unified evidence with review.
- `figure3_aaps_morphology_workflow.svg`: detailed morphology-method figure
  retaining the original AAPS design of image-prior inspection, adaptive method
  routing, metric selection, QC, database outputs, plots, and interpretation.
- `figure3_original_faithful_editable.svg`: faithful vertical reconstruction of
  the original five-panel morphology figure. The hidden
  `original-raster-reference` layer can be enabled for alignment checks; all
  diagram text, boxes, arrows, tables, metrics, and plots are native SVG objects.
- `figure4_viability_label_model_workflow.svg`: fluorescence-label QC and
  construction, paired training data, bright-field-only deployment path, and
  the editable ConvNeXt multi-scale regression architecture.

Microscopy panels should remain raster images in the final SVG. Statistical
plots should be regenerated from their CSV source as SVG/PDF rather than traced
from pixels. This preserves quantitative accuracy and editable axes, legends,
and curves.

Open the files in Inkscape, Illustrator, Affinity Designer, or recent
PowerPoint versions. Keep the legacy reference layer locked while rebuilding a
panel, then hide it before export.

Run `python build_legacy_svg_assets.py` to rebuild the canvases and components.
