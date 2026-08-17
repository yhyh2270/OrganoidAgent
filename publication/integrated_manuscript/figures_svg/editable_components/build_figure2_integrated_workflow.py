"""Generate the integrated morphology and viability AAPS workflow as native SVG."""

from pathlib import Path

from build_figure2_aaps_editable import (
    arrow,
    block,
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


OUT = Path(__file__).with_name("figure2_integrated_organoid_workflow.svg")
W, H = 1800, 1010


def header():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    <title>Integrated AAPS organoid morphology and viability workflow</title>
    <desc>Fully editable native-vector figure for the integrated OrganoidAgent workstation.</desc>
    <defs>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#47798a"/></marker>
      <style>
        text {{ font-family: Arial, Helvetica, sans-serif; fill:#17252b; }}
        .panel {{ fill:#f4f9fb; stroke:#9bbdca; stroke-width:3; }}
        .panel-label {{ font-size:32px; font-weight:700; }} .panel-title {{ font-size:27px; font-weight:500; }}
        .body {{ font-size:21px; }} .small {{ font-size:17px; }} .small-bold {{ font-size:17px; font-weight:700; }}
        .tiny {{ font-size:14px; }} .tiny-bold {{ font-size:15px; font-weight:700; }} .italic {{ font-size:18px; font-style:italic; }}
        .node {{ fill:#fff; stroke:#47798a; stroke-width:3; }} .soft-node {{ fill:#dceff4; stroke:#78a9b8; stroke-width:2.5; }}
        .morph-node {{ fill:#e5f6f1; stroke:#218b72; stroke-width:2.8; }} .viability-node {{ fill:#f1eafe; stroke:#7c4bc4; stroke-width:2.8; }}
        .qc-node {{ fill:#fff4d8; stroke:#c58b20; stroke-width:2.8; }} .result-node {{ fill:#e7f2fb; stroke:#347fb4; stroke-width:2.8; }}
        .block {{ fill:#fff; stroke:#47798a; stroke-width:3; }} .block-divider {{ stroke:#47798a; stroke-width:2.5; }} .block-title {{ font-size:18px; font-weight:700; }}
        .arrow {{ fill:none; stroke:#47798a; stroke-width:3; marker-end:url(#arrowhead); }} .line {{ fill:none; stroke:#47798a; stroke-width:3; }}
        .morph-line {{ fill:none; stroke:#218b72; stroke-width:4; marker-end:url(#arrowhead); }} .viability-line {{ fill:none; stroke:#7c4bc4; stroke-width:4; marker-end:url(#arrowhead); }}
        .dash {{ fill:none; stroke:#47798a; stroke-width:2.5; stroke-dasharray:7 6; }} .port {{ fill:#4197ad; stroke:#2f6573; stroke-width:1.5; }}
        .folder-fill {{ fill:#52a9b5; }} .icon-fill {{ fill:#f8fcfd; stroke:#47798a; stroke-width:3; }} .icon-line {{ fill:none; stroke:#47798a; stroke-width:3; }}
        .paper-back {{ fill:#dbe9ed; stroke:#7da5b2; stroke-width:2; }} .paper-mid {{ fill:#e8f2f4; stroke:#648e9b; stroke-width:2; }} .paper-front {{ fill:#f8fcfd; stroke:#47798a; stroke-width:2.5; }}
        .chart-bg {{ fill:#f8fcfd; stroke:#7da5b2; stroke-width:2; }} .chart-axis {{ fill:none; stroke:#6c8790; stroke-width:2; }}
      </style>
    </defs><rect width="100%" height="100%" fill="white"/>'''


def build():
    p = [header()]

    # A: shared workspace
    p.append(panel(12, 12, 510, 490, "A", "Project-aware organoid workspace", "A"))
    p.append(roundbox(38, 75, 300, 88, "node", "project-folder-box")); p.append(folder_icon(53, 83, .55))
    p.append(multiline(122, 108, ["Integrated organoid", "analysis project"], "body", dy=24))
    for yy, color, label in [(210,"#52a9b5","Raw images + metadata"),(252,"#d8a945","Models + checkpoints"),(294,"#9bbdca","Masks + features"),(336,"#9bbdca","Run logs + reports")]:
        p.append(f'<rect x="50" y="{yy-17}" width="23" height="22" rx="3" fill="{color}" stroke="#47798a" stroke-width="2"/>'); p.append(text(84, yy, label, "body"))
    p.append(text(350, 191, "Project context", "small"))
    names=["dataset_binding.aaps","image_qc.aaps","morphology_analysis.aaps","viability_detection.aaps","joint_report.aaps"]
    for i,name in enumerate(names):
        x=220+i*7; y=210+i*42
        p.append(roundbox(x,y,255,45,"node",f"workflow-{i+1}")); p.append(text(x+127,y+29,name,"small","middle"))
    p.append(text(260, 470, "Shared data, provenance, and artifacts", "body", "middle")); p.append('</g>')

    # B: top-down dual route
    p.append(panel(538, 12, 650, 490, "B", "Top-down dual-workflow design", "B"))
    p.append(text(1165, 76, "Task-oriented program", "italic", "end"))
    p.append(text(560,108,"1. Bind selected dataset or image scope","body"))
    p.append(text(560,141,"2. Inspect modality, quality, and experimental context","body"))
    p.append(text(560,174,"3. Route the requested analytical workflow","body"))
    p.append(roundbox(780,198,170,48,"qc-node","workflow-router")); p.append(text(865,229,"Workflow router","small-bold","middle"))
    p.append(f'<path d="M865 246 V270 H690 V294" class="line"/><path d="M865 270 H1040 V294" class="line"/>')
    p.append(roundbox(565,294,250,116,"morph-node","morphology-route")); p.append(text(690,324,"Morphological Analysis","small-bold","middle")); p.append(multiline(690,351,["Cellpose + signal recovery","instance merge + metrics"],"small","middle",23))
    p.append(roundbox(915,294,250,116,"viability-node","viability-route")); p.append(text(1040,324,"Viability Detection","small-bold","middle")); p.append(multiline(1040,351,["YOLO → SAM → ROI crop","checkpoint-1 inference"],"small","middle",23))
    p.append(f'<path d="M690 410 V433 H865" class="line"/><path d="M1040 410 V433 H865" class="line"/>')
    p.append(roundbox(740,426,250,50,"qc-node","review-and-report")); p.append(text(865,458,"QC → visualize → report","small-bold","middle"))
    p.append('</g>')

    # C: blocks
    p.append(panel(1204, 12, 584, 490, "C", "Reusable blocks and skills", "C"))
    p.append(block(1228,76,250,128,"image_qc",["image","metadata"],["qc_report","preview"],"image-qc"))
    p.append(block(1512,76,250,128,"bind_dataset",["dataset","selection"],["scope","manifest"],"bind"))
    p.append(block(1228,225,250,128,"segment_organoid",["image","qc_report"],["mask","overlay","instances"],"segment"))
    p.append(block(1512,225,250,128,"extract_viability_roi",["image","YOLO box"],["SAM mask","ROI crop"],"roi"))
    p.append(block(1228,374,250,98,"quantify_morphology",["mask"],["features"],"morphology"))
    p.append(block(1512,374,250,98,"predict_viability",["ROI crop"],["score"],"viability"))
    p.append('</g>')
    p.append(arrow(522,245,538,245,"A-to-B")); p.append(arrow(1188,245,1204,245,"B-to-C"))

    # D: constrained compilation
    p.append(panel(12, 520, 990, 475, "D", "Parser, policy, and agent-assisted compiler", "D"))
    p.append(file_icon(42,648)); p.append(multiline(76,750,[".aaps","workflow"],"small","middle",22))
    p.append(arrow(125,690,205,690)); p.append(roundbox(218,653,135,74,"soft-node","parser")); p.append(text(285,697,"parser","body","middle"))
    p.append(arrow(363,690,425,690)); p.append(stack_icon(438,645)); p.append(text(477,752,"structured IR","small","middle"))
    p.append(arrow(530,690,590,690)); p.append(roundbox(603,653,145,74,"soft-node","compiler")); p.append(text(675,686,"agent-assisted","small","middle")); p.append(text(675,709,"compiler","small","middle"))
    p.append(arrow(758,690,820,690)); p.append(roundbox(833,640,140,100,"result-node","execution-plan")); p.append(multiline(903,677,["validated","execution plan"],"small-bold","middle",23))
    p.append(roundbox(120,795,760,125,"node","execution-policy")); p.append(text(500,827,"Execution policy", "small-bold","middle"))
    policy=["dataset scope","allowed tools","model/checkpoint","fallback rules","output contract","review gate"]
    for i,item in enumerate(policy):
        x=155+(i%3)*240; y=861+(i//3)*36
        p.append(f'<circle cx="{x}" cy="{y-5}" r="5" fill="#47798a"/>'); p.append(text(x+14,y,item,"small"))
    p.append(f'<path d="M500 795 V762 H675 V727" class="dash"/>')
    p.append(text(500,958,"Deterministic tools where possible; bounded agent decisions where needed", "small","middle")); p.append('</g>')

    # E: dual execution and merged evidence
    p.append(panel(1018, 520, 770, 475, "E", "Controlled execution and unified outputs", "E"))
    # input
    for dx,dy,op in [(0,0,.55),(12,9,.75),(24,18,1)]:
        p.append(f'<g transform="translate({1048+dx} {650+dy})" opacity="{op}"><rect width="82" height="94" rx="5" fill="#d6dde0" stroke="#526b73" stroke-width="2"/><path d="M12 70 C4 49 24 25 41 35 C59 17 80 40 67 57 C81 77 52 88 39 76 C28 88 16 82 12 70Z" fill="#87969b"/></g>')
    p.append(text(1100,785,"Bright-field", "small","middle")); p.append(text(1100,807,"images", "small","middle"))
    p.append(arrow(1150,705,1200,705)); p.append(roundbox(1210,660,125,88,"qc-node","execution-qc")); p.append(multiline(1272,695,["QC + data","binding"],"small-bold","middle",22))
    # fork
    p.append(f'<path d="M1335 704 H1360 V620 H1390" class="morph-line"/><path d="M1360 704 V790 H1390" class="viability-line"/>')
    p.append(roundbox(1400,568,205,105,"morph-node","morph-execution")); p.append(text(1502,598,"Morphology route","small-bold","middle")); p.append(multiline(1502,626,["Cellpose / recovery","mask + features"],"small","middle",22))
    p.append(roundbox(1400,738,205,105,"viability-node","viability-execution")); p.append(text(1502,768,"Viability route","small-bold","middle")); p.append(multiline(1502,796,["YOLO + SAM crop","viability score"],"small","middle",22))
    # merge
    p.append(f'<path d="M1605 620 H1640 V705 H1660" class="morph-line"/><path d="M1605 790 H1640 V705 H1660" class="viability-line"/>')
    p.append(roundbox(1665,650,100,110,"result-node","joint-evidence")); p.append(multiline(1715,683,["Joint","evidence","package"],"small-bold","middle",22))
    p.append(text(1403,885,"Outputs", "small-bold","middle"))
    p.append(organoid_card(1260,900,["#26a6b7","#61be76","#844cc1"])); p.append(chart_icon(1380,905,"#218b72",0)); p.append(chart_icon(1470,905,"#7c4bc4",2))
    p.append(f'<g transform="translate(1590 884)"><rect width="54" height="82" rx="6" class="paper-back"/><rect x="-10" y="10" width="54" height="82" rx="6" class="paper-front"/><path d="M0 30 H34 M0 42 H34 M0 54 H24" class="icon-line"/></g>')
    p.append(text(1309,987,"overlay", "tiny","middle")); p.append(text(1416,987,"morphology", "tiny","middle")); p.append(text(1506,987,"viability", "tiny","middle")); p.append(text(1611,987,"report", "tiny","middle"))
    p.append('</g>')
    p.append(arrow(1002,755,1018,755,"D-to-E"))

    p.append('</svg>')
    return "\n".join(p)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
