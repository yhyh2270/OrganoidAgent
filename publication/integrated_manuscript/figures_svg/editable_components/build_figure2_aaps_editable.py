"""Generate a fully editable native-SVG reconstruction of the legacy AAPS figure."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("figure2_aaps_fully_editable.svg")
W, H = 1800, 940


def text(x, y, value, cls="body", anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{escape(value)}</text>'


def multiline(x, y, lines, cls="body", anchor="start", dy=27):
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else dy}">{escape(line)}</tspan>'
        for i, line in enumerate(lines)
    )
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{spans}</text>'


def roundbox(x, y, w, h, cls="node", rid=""):
    return f'<rect id="{rid}" x="{x}" y="{y}" width="{w}" height="{h}" rx="18" class="{cls}"/>'


def panel(x, y, w, h, label, title, rid):
    return (
        f'<g id="panel-{rid}">{roundbox(x, y, w, h, "panel", f"panel-{rid}-background")}'
        + text(x + 18, y + 38, label, "panel-label")
        + text(x + 58, y + 38, title, "panel-title")
    )


def arrow(x1, y1, x2, y2, rid=""):
    return f'<path id="{rid}" d="M{x1},{y1} L{x2},{y2}" class="arrow"/>'


def port(x, y):
    return f'<circle cx="{x}" cy="{y}" r="6" class="port"/>'


def file_icon(x, y, label=".aaps"):
    return f'''<g class="icon" transform="translate({x} {y})">
      <path d="M0 0 H48 L68 20 V88 H0 Z" class="icon-fill"/>
      <path d="M48 0 V20 H68" class="icon-line"/>
      {text(34, 48, label, "tiny-bold", "middle")}
      {text(34, 68, "script", "tiny", "middle")}
    </g>'''


def folder_icon(x, y, scale=1):
    return f'''<g class="icon" transform="translate({x} {y}) scale({scale})">
      <path d="M0 14 H32 L42 24 H92 V74 H0 Z" class="folder-fill"/>
      <path d="M0 14 H32 L42 24 H92 V74 H0 Z" class="icon-line"/>
      <path d="M4 29 H88" class="icon-line"/>
    </g>'''


def stack_icon(x, y):
    return f'''<g class="icon" transform="translate({x} {y})">
      <rect x="18" y="0" width="54" height="66" rx="9" class="paper-back"/>
      <rect x="9" y="8" width="54" height="66" rx="9" class="paper-mid"/>
      <rect x="0" y="16" width="54" height="66" rx="9" class="paper-front"/>
      <path d="M12 34 H42 M12 46 H42 M12 58 H36" class="icon-line"/>
    </g>'''


def chart_icon(x, y, color, mode=0):
    paths = ["M8 52 C28 43 35 20 62 11", "M8 14 C26 18 40 43 62 52", "M8 52 L26 34 L42 42 L62 13"]
    return f'''<g class="icon" transform="translate({x} {y})">
      <rect width="72" height="62" class="chart-bg"/>
      <path d="M8 8 V54 H66" class="chart-axis"/>
      <path d="{paths[mode]}" fill="none" stroke="{color}" stroke-width="5"/>
    </g>'''


def organoid_card(x, y, colors):
    blobs = "".join(
        f'<path d="{d}" fill="{c}" stroke="#254f5f" stroke-width="2"/>'
        for d, c in zip(
            [
                "M17 46 C6 29 18 8 38 13 C54 0 77 13 71 32 C84 51 59 68 42 57 C28 70 10 61 17 46Z",
                "M50 20 C62 8 84 17 82 35 C94 47 82 66 65 60 C52 70 37 57 42 43 C34 32 39 24 50 20Z",
                "M13 22 C19 8 37 7 44 19 C55 24 51 43 39 47 C31 58 11 49 14 35 C5 31 6 25 13 22Z",
            ],
            colors,
        )
    )
    return f'<g transform="translate({x} {y})"><rect width="98" height="76" rx="5" fill="#101820"/>{blobs}</g>'


def block(x, y, w, h, title_value, inputs, outputs, rid):
    s = [f'<g id="block-{rid}">', roundbox(x, y, w, h, "block", f"block-{rid}-background")]
    s.append(f'<path d="M{x},{y+38} H{x+w}" class="block-divider"/>')
    s.append(text(x + w / 2, y + 27, title_value, "block-title", "middle"))
    max_rows = max(len(inputs), len(outputs))
    for i in range(max_rows):
        yy = y + 63 + i * 27
        if i < len(inputs):
            s.extend([port(x, yy - 6), text(x + 16, yy, inputs[i], "small")])
        if i < len(outputs):
            s.extend([port(x + w, yy - 6), text(x + w - 16, yy, outputs[i], "small", "end")])
    s.append('</g>')
    return "".join(s)


def build():
    p = []
    p.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    <title>Fully editable AAPS structured analytical workflow</title>
    <desc>Native vector reconstruction. Every panel, label, box, connector, port, and icon is editable.</desc>
    <defs>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#47798a"/></marker>
      <style>
        text {{ font-family: Arial, Helvetica, sans-serif; fill:#17252b; }}
        .panel {{ fill:#eef8fb; stroke:#9bbdca; stroke-width:3; }}
        .panel-label {{ font-size:32px; font-weight:700; }} .panel-title {{ font-size:28px; font-weight:500; }}
        .body {{ font-size:22px; }} .small {{ font-size:18px; }} .small-bold {{ font-size:18px; font-weight:700; }}
        .tiny {{ font-size:14px; }} .tiny-bold {{ font-size:15px; font-weight:700; }} .italic {{ font-size:19px; font-style:italic; }}
        .node {{ fill:#f9fdfe; stroke:#47798a; stroke-width:3; }} .soft-node {{ fill:#d9eef3; stroke:#78a9b8; stroke-width:2.5; }}
        .block {{ fill:#f8fcfd; stroke:#47798a; stroke-width:3; }} .block-divider {{ stroke:#47798a; stroke-width:2.5; }} .block-title {{ font-size:20px; font-weight:700; }}
        .arrow {{ fill:none; stroke:#47798a; stroke-width:3; marker-end:url(#arrowhead); }} .line {{ fill:none; stroke:#47798a; stroke-width:3; }}
        .dash {{ fill:none; stroke:#47798a; stroke-width:2.5; stroke-dasharray:7 6; }} .port {{ fill:#4197ad; stroke:#2f6573; stroke-width:1.5; }}
        .folder-fill {{ fill:#52a9b5; }} .icon-fill {{ fill:#f8fcfd; stroke:#47798a; stroke-width:3; }} .icon-line {{ fill:none; stroke:#47798a; stroke-width:3; }}
        .paper-back {{ fill:#dbe9ed; stroke:#7da5b2; stroke-width:2; }} .paper-mid {{ fill:#e8f2f4; stroke:#648e9b; stroke-width:2; }} .paper-front {{ fill:#f8fcfd; stroke:#47798a; stroke-width:2.5; }}
        .chart-bg {{ fill:#f8fcfd; stroke:#7da5b2; stroke-width:2; }} .chart-axis {{ fill:none; stroke:#6c8790; stroke-width:2; }}
      </style>
    </defs><rect width="100%" height="100%" fill="white"/>''')

    # Panel A
    p.append(panel(12, 12, 530, 470, "A", "Project-aware AAPS workspace", "A"))
    p.append(roundbox(38, 78, 300, 92, "node", "project-folder-box")); p.append(folder_icon(53, 88, .55))
    p.append(multiline(122, 112, ["DEO organoid", "analysis project"], "body", dy=25))
    for yy, color, label in [(222,"#50a8b5","Data"),(265,"#d8a945","Artifacts"),(308,"#9bbdca","Run log"),(351,"#9bbdca","Report")]:
        p.append(f'<rect x="51" y="{yy-17}" width="23" height="22" rx="3" fill="{color}" stroke="#47798a" stroke-width="2"/>'); p.append(text(84, yy, label, "body"))
    p.append(text(350, 197, "Project context", "small"))
    workflow_names=["dataset_registration.aaps","image_qc.aaps","segmentation_routing.aaps","metric_quantification.aaps","report_generation.aaps"]
    for i,name in enumerate(workflow_names):
        x=224+i*7; y=224+i*37
        p.append(roundbox(x,y,270,43,"node",f"workflow-{i+1}")); p.append(text(x+135,y+28,name,"small","middle"))
    p.append(text(279, 450, "Many .aaps workflows", "body", "middle")); p.append('</g>')

    # Panel B
    p.append(panel(558, 12, 630, 470, "B", "Top-down GUI program design", "B"))
    p.append(text(1164, 81, "Top-down program", "italic", "end"))
    steps=["1. Register raw images and metadata","2. Inspect image quality and experimental context","3. Route segmentation method","4. Verify mask and overlay","5. Quantify growth, differentiation, and fusion metrics","6. Summarize biological conclusions"]
    for i,s in enumerate(steps[:3]): p.append(text(580,112+i*36,s,"body"))
    p.append(roundbox(680,210,430,50,"soft-node","loop-node")); p.append(text(895,242,"↻  for each image        QC and method routing","small","middle"))
    p.append(f'<path d="M700 260 V350 H735" class="line"/>')
    p.append(roundbox(735,278,410,54,"soft-node","low-confidence-node")); p.append(text(940,312,"⚡ low confidence → human review / correction","small","middle"))
    p.append(roundbox(735,342,410,62,"soft-node","route-node")); p.append(multiline(940,369,["⚡ route: Cellpose / thresholding / signal recovery /","prompt-guided correction"],"small","middle",22))
    for i,s in enumerate(steps[3:]): p.append(text(580,414+i*29,s,"body"))
    p.append('</g>')

    # Panel C
    p.append(panel(1204, 12, 584, 470, "C", "Reusable blocks and skills", "C"))
    p.append(block(1232,82,245,145,"image_qc",["image","metadata"],["qc_report","preview"],"image-qc"))
    p.append(block(1515,82,245,145,"segment_organoid",["image","qc_report"],["mask","overlay","instances"],"segment"))
    p.append(block(1232,265,245,145,"quantify_metrics",["mask","image"],["area","darkness_P90","fusion_score"],"metrics"))
    p.append(block(1515,265,245,145,"review_overlay",["overlay","confidence"],["approval","correction"],"review"))
    p.append(text(1496,441,"Block contract = inputs + outputs + tools", "body","middle")); p.append(text(1496,470,"+ agent + validation + artifacts", "body","middle")); p.append('</g>')

    # Inter-panel arrows top
    p.append(arrow(542,235,558,235,"A-to-B")); p.append(arrow(1188,235,1204,235,"B-to-C"))

    # Panel D
    p.append(panel(12, 500, 1015, 425, "D", "Parser, grammar, and agent-based compiler", "D"))
    p.append(file_icon(45,620)); p.append(multiline(79,724,["example","code"],"small","middle",22))
    p.append('<rect x="205" y="560" width="66" height="56" rx="12" class="dash"/>')
    p.append(arrow(281,588,352,588)); p.append(stack_icon(367,543)); p.append(arrow(457,588,515,588))
    p.append(roundbox(528,553,76,70,"node","compiler-icon")); p.append(arrow(614,588,700,588)); p.append(stack_icon(716,543));
    p.append(roundbox(810,660,145,58,"soft-node","execution-plan")); p.append(text(882,685,"execution", "small","middle")); p.append(text(882,707,"plan", "small","middle"))
    labels=[(205,"parser","deterministic parse"),(365,"structured IR","recursive module filling"),(515,"compiler",""),(700,"resolved workflow","agent-assisted generation")]
    for x,title,sub in labels:
        p.append(roundbox(x,660,145 if x!=365 else 155,58,"soft-node",title.replace(' ', '-'))); p.append(text(x+(72 if x!=365 else 77),695,title,"small","middle"));
        if sub: p.append(text(x+(72 if x!=365 else 77),742,sub,"small","middle"))
    p.append(text(938,540,"top-down recursive", "small","end")); p.append(text(938,564,"fill-in motif", "small","end"))
    p.append(f'<path d="M870 623 V647" class="arrow"/>')
    p.append(f'<path d="M540 718 V770 H895 V718" class="dash"/>')
    miss=[("Missing block", "generate reusable", ".aaps block"),("Missing script","write Python /","shell script"),("Missing tool","report setup or","select fallback"),("Missing dependency","prepare safe","setup prompt"),("Ambiguous result","ask human","review")]
    for i,(head,l1,l2) in enumerate(miss):
        x=28+i*198; p.append(text(x+84,810,head,"small-bold","middle")); p.append(text(x+84,838,l1,"small","middle")); p.append(text(x+84,862,l2,"small","middle"))
    p.append('</g>')

    # Panel E
    p.append(panel(1044, 500, 744, 425, "E", "Controlled execution and biological outputs", "E"))
    # vector input image stack
    for dx,dy,op in [(0,0,.55),(14,10,.72),(28,20,1)]:
        p.append(f'<g transform="translate({1080+dx} {610+dy})" opacity="{op}"><rect width="94" height="108" rx="5" fill="#d6dde0" stroke="#526b73" stroke-width="2"/><path d="M15 78 C4 54 27 27 47 39 C68 18 92 45 76 63 C93 86 58 99 45 84 C31 98 19 91 15 78Z" fill="#87969b"/><circle cx="66" cy="33" r="10" fill="#aab5b9"/></g>')
    p.append(arrow(1214,674,1270,674)); p.append('<circle cx="1302" cy="674" r="34" fill="#d8f2f1" stroke="#47798a" stroke-width="3"/><path d="M1285 674 L1298 687 L1320 656" fill="none" stroke="#47798a" stroke-width="4"/>')
    p.append(arrow(1338,674,1388,674)); p.append(organoid_card(1400,592,["#26a6b7","#61be76","#844cc1"])); p.append(organoid_card(1421,614,["#28a6b8","#6abb73","#9a4fbd"])); p.append(organoid_card(1442,636,["#39a9bc","#73bf68","#9650b8"]))
    p.append(arrow(1545,674,1590,674)); p.append(chart_icon(1600,574,"#63b7c7",0)); p.append(chart_icon(1600,648,"#58a8c8",1)); p.append(chart_icon(1600,722,"#db932d",2)); p.append(arrow(1675,674,1710,674))
    p.append('<g transform="translate(1715 585)"><rect width="54" height="142" rx="8" class="paper-back"/><rect x="-10" y="10" width="54" height="142" rx="8" class="paper-mid"/><rect x="-20" y="20" width="54" height="142" rx="8" class="paper-front"/><rect x="-10" y="34" width="34" height="16" fill="#d6eef2"/><path d="M-10 65 H24 M-10 77 H24 M-10 89 H12" class="icon-line"/><rect x="-10" y="103" width="13" height="13" fill="#e8b04e"/><circle cx="18" cy="110" r="8" fill="#70b68b"/><path d="M-10 141 L1 129 L10 135 L24 119" fill="none" stroke="#4ba4b8" stroke-width="3"/></g>')
    p.append(multiline(1142,782,["Heterogeneous","DEO bright-field","images"],"small","middle",22))
    p.append(multiline(1302,782,["QC:","brightness, blur,","density, object scale"],"small","middle",22))
    p.append(multiline(1467,782,["Segmentation","and metrics"],"small","middle",22))
    p.append(multiline(1636,782,["biological","metrics"],"small","middle",22))
    p.append(multiline(1741,782,["Artifacts","and report"],"small","middle",22))
    p.append(text(1416,895,"Validation and human review", "body","middle")); p.append('</g>')
    p.append(arrow(1027,708,1044,708,"D-to-E"))

    p.append('</svg>')
    return "\n".join(p)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
