"""Build the publication-ready four-panel integrated OrganoidAgent workflow."""

from pathlib import Path

from build_figure2_aaps_editable import (
    arrow,
    chart_icon,
    file_icon,
    folder_icon,
    multiline,
    organoid_card,
    panel,
    roundbox,
    stack_icon,
    text,
)


OUT = Path(__file__).with_name("figure2_integrated_workflow_4panel.svg")
W, H = 2000, 1100


def header():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <title>OrganoidAgent integrated morphology and viability workflow</title>
  <desc>Editable four-panel workflow: data context, constrained orchestration, parallel analyses, and unified results.</desc>
  <defs>
    <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#496f7d"/></marker>
    <marker id="morph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#17846b"/></marker>
    <marker id="viability-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#7650b5"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill:#17252b; }}
      .panel {{ fill:#f7fafb; stroke:#aac3cc; stroke-width:3; }}
      .panel-label {{ font-size:40px; font-weight:700; }} .panel-title {{ font-size:34px; font-weight:600; }}
      .body {{ font-size:28px; }} .small {{ font-size:24px; }} .small-bold {{ font-size:24px; font-weight:700; }}
      .tiny {{ font-size:21px; }} .tiny-bold {{ font-size:21px; font-weight:700; }} .italic {{ font-size:23px; font-style:italic; }}
      .node {{ fill:#fff; stroke:#496f7d; stroke-width:2.8; }} .shared {{ fill:#e9f3f6; stroke:#4f899b; stroke-width:2.8; }}
      .morph {{ fill:#e9f3f6; stroke:#4f899b; stroke-width:3; }} .viability {{ fill:#e9f3f6; stroke:#4f899b; stroke-width:3; }}
      .review {{ fill:#e9f3f6; stroke:#4f899b; stroke-width:3; }} .result {{ fill:#e9f3f6; stroke:#4f899b; stroke-width:3; }}
      .arrow {{ fill:none; stroke:#496f7d; stroke-width:3; marker-end:url(#arrowhead); }}
      .morph-arrow {{ fill:none; stroke:#496f7d; stroke-width:3.5; marker-end:url(#arrowhead); }}
      .viability-arrow {{ fill:none; stroke:#496f7d; stroke-width:3.5; marker-end:url(#arrowhead); }}
      .line {{ fill:none; stroke:#496f7d; stroke-width:3; }} .dash {{ fill:none; stroke:#718f9a; stroke-width:2.5; stroke-dasharray:8 7; }}
      .folder-fill {{ fill:#55aab7; }} .icon-fill {{ fill:#fff; stroke:#496f7d; stroke-width:3; }} .icon-line {{ fill:none; stroke:#496f7d; stroke-width:3; }}
      .paper-back {{ fill:#dbe9ed; stroke:#7da5b2; stroke-width:2; }} .paper-mid {{ fill:#e8f2f4; stroke:#648e9b; stroke-width:2; }} .paper-front {{ fill:#fff; stroke:#496f7d; stroke-width:2.5; }}
      .chart-bg {{ fill:#fff; stroke:#7da5b2; stroke-width:2; }} .chart-axis {{ fill:none; stroke:#6c8790; stroke-width:2; }}
    </style>
  </defs><rect width="100%" height="100%" fill="white"/>'''


def image_card(x, y, label, accent="#4f899b"):
    return f'''<g><rect x="{x}" y="{y}" width="145" height="112" rx="10" fill="#dce5e8" stroke="{accent}" stroke-width="2.5"/>
      <path d="M{x+21} {y+78} C{x+8} {y+51} {x+32} {y+24} {x+58} {y+35} C{x+81} {y+13} {x+127} {y+40} {x+111} {y+67} C{x+128} {y+93} {x+86} {y+105} {x+68} {y+86} C{x+47} {y+105} {x+27} {y+94} {x+21} {y+78}Z" fill="#8c9b9f"/>
      <circle cx="{x+103}" cy="{y+31}" r="12" fill="#b3bec1"/><text x="{x+72}" y="{y+139}" class="small" text-anchor="middle">{label}</text></g>'''


def step_box(x, y, w, title_value, lines, cls="shared", rid=""):
    h = 82 + max(0, len(lines) - 1) * 22
    parts = [f'<g id="{rid}">', roundbox(x, y, w, h, cls, f"{rid}-background"), text(x+w/2, y+31, title_value, "small-bold", "middle")]
    parts.append(multiline(x+w/2, y+58, lines, "small", "middle", 22))
    parts.append('</g>')
    return "".join(parts), h


def build():
    p = [header()]

    # A — inputs and context
    p.append(panel(18, 18, 620, 450, "A", "Data visualization and input", "A"))
    p.append(image_card(48, 92, "Single image"))
    p.append(image_card(230, 92, "Batch / time series"))
    p.append('<g id="dataset-metadata-input"><rect x="412" y="92" width="185" height="112" rx="10" fill="#ffffff" stroke="#4f899b" stroke-width="2.5"/><path d="M438 118 H567 M438 142 H567 M438 166 H567" stroke="#9bbdca" stroke-width="2"/><path d="M470 105 V190 M525 105 V190" stroke="#9bbdca" stroke-width="2"/><rect x="438" y="105" width="129" height="85" fill="none" stroke="#496f7d" stroke-width="2.5"/><text x="504" y="231" class="small" text-anchor="middle">Metadata</text><text x="504" y="253" class="tiny" text-anchor="middle">condition • time • path</text></g>')
    p.append(roundbox(48,286,549,130,"node","project-workspace")); p.append(folder_icon(70,305,.55))
    p.append(text(140,325,"Data Visualization","body"))
    items=[("Dataset index",160,360),("Image preview",360,360),("Selected files",160,393),("Experimental metadata",360,393)]
    for label,x,y in items:
        p.append(f'<circle cx="{x-15}" cy="{y-6}" r="5" fill="#4f899b"/>'); p.append(text(x,y,label,"small"))
    p.append('</g>')

    # B — agent orchestration
    p.append(panel(658, 18, 1324, 450, "B", "Workflow selection and execution", "B"))
    p.append(roundbox(692,88,250,98,"node","user-request")); p.append(text(817,120,"Request","small-bold","middle")); p.append(multiline(817,148,["selected data","+ instruction"],"small","middle",24))
    p.append(arrow(942,137,1000,137,"request-to-bind"))
    p.append(roundbox(1010,88,240,98,"shared","data-binding")); p.append(text(1130,120,"Bind data","small-bold","middle")); p.append(multiline(1130,148,["dataset / images","condition / time"],"small","middle",24))
    p.append(arrow(1250,137,1308,137,"bind-to-parse"))
    p.append(roundbox(1318,88,270,98,"shared","workflow-router")); p.append(text(1453,120,"Route workflow","small-bold","middle")); p.append(multiline(1453,148,["AAPS morphology","or viability runner"],"small","middle",24))
    p.append(arrow(1588,137,1646,137,"parse-to-plan"))
    p.append(roundbox(1656,88,285,98,"result","validated-plan")); p.append(text(1798,120,"Validate run","small-bold","middle")); p.append(multiline(1798,148,["tools + models","output contract"],"small","middle",24))
    # policy strip
    p.append(roundbox(692,225,1249,155,"review","execution-policy")); p.append(text(1316,258,"Execution policy and reproducibility controls","body","middle"))
    policy=[("Data scope",""),("Allowed tools",""),("Model identity",""),("Fallback rules",""),("QC gate",""),("Human review","")]
    for i,(title_value,sub) in enumerate(policy):
        x=720+i*202
        p.append(roundbox(x,282,174,70,"node",f"policy-{i+1}")); p.append(text(x+87,325,title_value,"tiny-bold","middle"))
    p.append(text(1316,421,"Deterministic processing where possible; bounded and auditable agent decisions.","small","middle"))
    p.append('</g>')
    p.append(arrow(638,242,658,242,"A-to-B"))

    # C — dual workflows
    p.append(panel(18, 490, 1215, 590, "C", "Parallel analysis workflows", "C"))
    p.append(text(1165,548,"Independent or joint execution","italic","end"))
    p.append(roundbox(52,560,190,100,"shared","shared-input")); p.append(text(147,594,"QC-passed","small-bold","middle")); p.append(multiline(147,621,["bright-field image","+ execution plan"],"small","middle",21))
    # Morphology lane
    p.append(text(286,588,"I. Morphological Analysis — AAPS / Codex-orchestrated","body")); p.append(f'<rect x="286" y="604" width="879" height="172" rx="22" class="morph"/>')
    morph=[("Cellpose",["multiscale","segmentation"]),("Recovery + merge",["large objects","instance mask"]),("Morphology metrics",["area • roundness","darkness • fusion"])]
    mx=[320,600,880]
    for i,((title_value,lines),x) in enumerate(zip(morph,mx)):
        s,h=step_box(x,635,245,title_value,lines,"node",f"morph-step-{i+1}"); p.append(s)
        if i<len(morph)-1: p.append(f'<path d="M{x+245} {684} H{mx[i+1]-10}" class="morph-arrow"/>')
    # Viability lane
    p.append(text(286,829,"II. Viability Detection — deterministic inference runner","body")); p.append(f'<rect x="286" y="845" width="879" height="172" rx="22" class="viability"/>')
    viability=[("YOLO localization",["organoid box"]),("SAM ROI extraction",["mask • overlay","model crop"]),("Viability inference",["checkpoint-1","ConvNeXt score"])]
    for i,((title_value,lines),x) in enumerate(zip(viability,mx)):
        s,h=step_box(x,876,245,title_value,lines,"node",f"viability-step-{i+1}"); p.append(s)
        if i<len(viability)-1: p.append(f'<path d="M{x+245} {925} H{mx[i+1]-10}" class="viability-arrow"/>')
    p.append(f'<path d="M242 610 H265 V684 H300" class="morph-arrow"/><path d="M265 684 V925 H300" class="viability-arrow"/>')
    p.append(text(1134,1052,"Both routes operate on bright-field input images","tiny-bold","end"))
    p.append('</g>')
    p.append(f'<path d="M1320 468 V480 H630 V490" class="arrow"/>')

    # D — unified results
    p.append(panel(1253, 490, 729, 590, "D", "Unified results and review", "D"))
    p.append(roundbox(1285,558,665,112,"result","evidence-package")); p.append(text(1617,596,"Unified evidence package","body","middle")); p.append(multiline(1617,630,["source • model • parameters","outputs • QC • provenance"],"small","middle",25))
    # visual outputs row
    p.append(organoid_card(1310,700,["#4f899b","#82aab5","#6d8790"])); p.append(text(1359,805,"Overlay","small","middle"))
    p.append(chart_icon(1460,708,"#4f899b",0)); p.append(text(1496,805,"Metrics","small","middle"))
    p.append(roundbox(1580,694,150,96,"result","rank-card")); p.append(text(1655,725,"Viability","tiny-bold","middle")); p.append(text(1655,754,"score + rank","small","middle")); p.append(text(1655,805,"Prediction","small","middle"))
    p.append(f'<g transform="translate(1840 692)"><rect width="70" height="100" rx="7" class="paper-back"/><rect x="-13" y="12" width="70" height="100" rx="7" class="paper-front"/><path d="M0 34 H43 M0 48 H43 M0 62 H30" class="icon-line"/><path d="M0 91 L13 78 L24 84 L43 68" fill="none" stroke="#4f899b" stroke-width="3"/></g>'); p.append(text(1868,818,"Report","small","middle"))
    p.append(roundbox(1290,845,425,155,"node","results-workstation")); p.append(text(1502,878,"Results Visualization","body","middle")); p.append(multiline(1502,910,["inspect image-level evidence","compare conditions and time points","export SVG / CSV / JSON / report"],"small","middle",24))
    p.append(roundbox(1742,845,200,155,"review","human-review")); p.append(text(1842,882,"Human review","small-bold","middle")); p.append(multiline(1842,914,["accept","flag uncertainty","revise workflow"],"small","middle",24))
    p.append(f'<path d="M1715 923 H1732" class="arrow"/>')
    # feedback loop
    p.append(f'<path d="M1842 845 V830 H1960 V530 H1935" class="dash"/>'); p.append(text(1898,555,"feedback", "tiny", "middle"))
    p.append('</g>')
    p.append(arrow(1233,785,1253,785,"C-to-D"))

    p.append('</svg>')
    return "\n".join(p)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
