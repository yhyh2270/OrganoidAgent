"""Build editable SVG canvases and reusable vector workflow components.

Legacy paper figures are flattened raster pages. The generated canvases keep the
page image as a replaceable layer; reusable workflow diagrams are reconstructed
as native SVG objects in separate files.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY = Path(r"F:\新建文件夹\aaps-organoid-20260509_155431")
ASSETS = ROOT / "assets"
CANVASES = ROOT / "legacy_canvases"
COMPONENTS = ROOT / "editable_components"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
  width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <title>{html.escape(title)}</title>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
    </marker>
    <style>
      .label {{ font: 600 28px Arial, Helvetica, sans-serif; fill: #0f172a; }}
      .body {{ font: 22px Arial, Helvetica, sans-serif; fill: #334155; }}
      .small {{ font: 18px Arial, Helvetica, sans-serif; fill: #475569; }}
      .box {{ fill: #ffffff; stroke: #334155; stroke-width: 2.5; rx: 18; }}
      .arrow {{ fill: none; stroke: #334155; stroke-width: 3; marker-end: url(#arrow); }}
    </style>
  </defs>
'''


def build_canvases() -> list[dict]:
    manifest = json.loads((LEGACY / "figures_manifest.json").read_text(encoding="utf-8"))
    records = []
    ASSETS.mkdir(parents=True, exist_ok=True)
    CANVASES.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        png_name = Path(item["png"]).name
        shutil.copy2(LEGACY / item["png"], ASSETS / png_name)
        width, height = item["width_px"], item["height_px"]
        svg_name = Path(png_name).with_suffix(".svg").name
        svg = svg_header(width, height, f"Editable canvas for {png_name}") + f'''
  <!-- The legacy page is intentionally isolated as one replaceable layer. -->
  <g id="legacy-raster-reference" opacity="1">
    <image x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none"
      xlink:href="../assets/{html.escape(png_name)}"/>
  </g>
  <!-- Add reconstructed vector panels above this layer and hide the reference when ready. -->
  <g id="editable-vector-content"/>
</svg>
'''
        write(CANVASES / svg_name, svg)
        records.append({**item, "editable_canvas": f"legacy_canvases/{svg_name}"})
    write(ROOT / "conversion_manifest.json", json.dumps(records, ensure_ascii=False, indent=2))
    return records


def data_pipeline_svg() -> str:
    w, h = 1600, 420
    nodes = [
        (50, "Raw bright-field\nimages", "#eff6ff", "#2563eb"),
        (350, "Image QC and\nmetadata validation", "#ecfeff", "#0891b2"),
        (650, "Canonical dataset\nand provenance", "#f0fdf4", "#16a34a"),
        (950, "Morphology and\nviability workflows", "#faf5ff", "#9333ea"),
    ]
    out = [svg_header(w, h, "Editable data-to-analysis workflow")]
    out.append('<rect width="1600" height="420" fill="white"/>')
    for i, (x, text, fill, stroke) in enumerate(nodes):
        out.append(f'<g id="step-{i+1}"><rect x="{x}" y="115" width="240" height="150" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        lines = text.split("\n")
        for j, line in enumerate(lines):
            out.append(f'<text x="{x+120}" y="{172+j*36}" text-anchor="middle" class="body">{html.escape(line)}</text>')
        out.append('</g>')
        if i < len(nodes) - 1:
            out.append(f'<path d="M {x+245} 190 H {x+292}" class="arrow"/>')
    out.append('<path d="M 1190 190 H 1280 V 72 H 1350" class="arrow"/>')
    out.append('<path d="M 1280 190 V 190 H 1350" class="arrow"/>')
    out.append('<path d="M 1280 190 V 308 H 1350" class="arrow"/>')
    for y, label, fill, stroke in [(35, "Density", "#eff6ff", "#2563eb"), (153, "Alginate", "#f0fdf4", "#16a34a"), (271, "Y-27632", "#fff1f2", "#e11d48")]:
        out.append(f'<g><rect x="1350" y="{y}" width="200" height="74" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="3"/><text x="1450" y="{y+47}" text-anchor="middle" class="body">{label}</text></g>')
    out.append('</svg>')
    return "\n".join(out)


def segmentation_pipeline_svg() -> str:
    w, h = 1800, 620
    out = [svg_header(w, h, "Editable multiscale segmentation workflow"), '<rect width="1800" height="620" fill="white"/>']
    boxes = [
        (45, 220, 230, 120, "Raw TIFF", "bright-field image", "#eff6ff", "#2563eb"),
        (340, 220, 250, 120, "Normalize", "grayscale + CLAHE", "#f8fafc", "#64748b"),
        (655, 220, 250, 120, "Hybrid signal", "foreground + residual", "#f0fdf4", "#16a34a"),
        (970, 70, 250, 120, "Cellpose", "multiscale branches", "#fff7ed", "#ea580c"),
        (970, 250, 250, 120, "Support mask", "signal thresholds", "#f0fdf4", "#16a34a"),
        (970, 430, 250, 120, "Large recovery", "components + watershed", "#faf5ff", "#9333ea"),
        (1305, 220, 260, 120, "Candidate scoring", "overlap-aware merge", "#eef2ff", "#4f46e5"),
        (1630, 220, 130, 120, "Final", "instances", "#ecfeff", "#0891b2"),
    ]
    for i, (x, y, bw, bh, title, subtitle, fill, stroke) in enumerate(boxes):
        out.append(f'<g id="node-{i+1}"><rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="3"/><text x="{x+bw/2}" y="{y+49}" text-anchor="middle" class="body">{html.escape(title)}</text><text x="{x+bw/2}" y="{y+82}" text-anchor="middle" class="small">{html.escape(subtitle)}</text></g>')
    paths = [
        "M 275 280 H 330", "M 590 280 H 645", "M 905 280 H 935 V 130 H 960",
        "M 905 280 H 960", "M 905 280 H 935 V 490 H 960", "M 1220 130 H 1260 V 280 H 1295",
        "M 1220 310 H 1295", "M 1220 490 H 1260 V 310 H 1295", "M 1565 280 H 1620",
    ]
    for path in paths:
        out.append(f'<path d="{path}" class="arrow"/>')
    out.append('<text x="45" y="65" class="label">Multiscale organoid instance segmentation</text>')
    out.append('<text x="45" y="100" class="small">All nodes, labels, colors, and connectors are editable SVG objects.</text>')
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    records = build_canvases()
    write(COMPONENTS / "figure1_data_pipeline.svg", data_pipeline_svg())
    write(COMPONENTS / "figure2_segmentation_pipeline.svg", segmentation_pipeline_svg())
    print(f"Created {len(records)} SVG canvases and 2 native-vector components in {ROOT}")


if __name__ == "__main__":
    main()
