"""Build a publication-ready vector/raster hybrid AAPS morphology figure."""

from pathlib import Path

from PIL import Image

from build_figure2_aaps_editable import arrow, multiline, panel, roundbox, text


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
OUT = HERE / "figure3_aaps_morphology_workflow.svg"
PREVIEW_SOURCE = ASSETS / "morphology_workflow_reference.png"
W, H = 1900, 1260


def prepare_assets():
    im = Image.open(PREVIEW_SOURCE).convert("RGB")
    crops = {
        "morph_heterogeneous.png": (55, 0, 495, 245),
        "morph_segmentation_qc.png": (675, 480, 958, 825),
        "morph_outputs.png": (300, 1230, 690, 1585),
    }
    for name, box in crops.items():
        im.crop(box).save(ASSETS / name, quality=95)


def header():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    <title>AAPS morphology analysis workflow</title>
    <defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#496f7d"/></marker>
    <style>
    text{{font-family:Arial,Helvetica,sans-serif;fill:#17252b}} .panel{{fill:#f7fafb;stroke:#aac3cc;stroke-width:3}}
    .panel-label{{font-size:40px;font-weight:700}} .panel-title{{font-size:33px;font-weight:600}} .body{{font-size:27px}} .small{{font-size:23px}} .small-bold{{font-size:23px;font-weight:700}} .tiny{{font-size:19px}} .tiny-bold{{font-size:19px;font-weight:700}}
    .node{{fill:#fff;stroke:#4f899b;stroke-width:3}} .soft-node{{fill:#e9f3f6;stroke:#4f899b;stroke-width:3}} .warn{{fill:#fff7f2;stroke:#cf6540;stroke-width:3;stroke-dasharray:8 6}}
    .arrow{{fill:none;stroke:#496f7d;stroke-width:3;marker-end:url(#arrowhead)}} .line{{fill:none;stroke:#496f7d;stroke-width:3}} .dash{{fill:none;stroke:#4f899b;stroke-width:3;stroke-dasharray:7 6;marker-end:url(#arrowhead)}}
    </style></defs><rect width="100%" height="100%" fill="white"/>'''


def mini_mask(x, y, good=True):
    blobs = [(25,28,18),(55,22,17),(82,35,19),(40,58,20),(72,65,17)]
    s=[f'<g transform="translate({x} {y})"><rect width="120" height="92" rx="8" fill="#d9e1e4" stroke="#7d969f" stroke-width="2"/>']
    for i,(cx,cy,r) in enumerate(blobs):
        fill=["#7299a5","#93b1ba","#5e8794","#a8c0c7","#7aa1ad"][i]
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="#496f7d" stroke-width="2"/>')
    s.append(text(105,18,"✓" if good else "×","small-bold","middle",f'fill="{"#23845f" if good else "#c43d3d"}"'))
    s.append('</g>'); return ''.join(s)


def metric_card(x,y,title_value,subtitle,kind):
    icon=""
    if kind=="growth": icon=f'<circle cx="{x+72}" cy="{y+91}" r="38" fill="none" stroke="#4f899b" stroke-width="3"/>'+''.join(f'<circle cx="{x+48+(i%3)*24}" cy="{y+72+(i//3)*23}" r="5" fill="#4f899b"/>' for i in range(9))
    elif kind=="dark": icon=''.join(f'<rect x="{x+36+i*10}" y="{y+125-h}" width="8" height="{h}" fill="#4f899b"/>' for i,h in enumerate([18,30,45,60,72,58,42,28]))
    elif kind=="shape": icon=f'<path d="M{x+35} {y+120} C{x+20} {y+80} {x+50} {y+55} {x+88} {y+62} C{x+124} {y+45} {x+138} {y+86} {x+118} {y+112} C{x+92} {y+140} {x+58} {y+132} {x+35} {y+120}Z" fill="#dcecef" stroke="#4f899b" stroke-width="3"/>'
    else: icon=f'<circle cx="{x+52}" cy="{y+96}" r="25" fill="#dcecef" stroke="#4f899b" stroke-width="3"/><circle cx="{x+110}" cy="{y+96}" r="25" fill="#dcecef" stroke="#4f899b" stroke-width="3"/><path d="M{x+76} {y+96} H{x+85}" class="arrow"/>'
    return f'<g>{roundbox(x,y,155,160,"node",kind)}{text(x+77,y+31,title_value,"small-bold","middle")}{text(x+77,y+56,subtitle,"tiny","middle")}{icon}</g>'


def build():
    prepare_assets(); p=[header()]
    # A motivation
    p.append(panel(16,16,1868,235,"A","Heterogeneous images challenge unconstrained analysis","A"))
    p.append('<image x="55" y="68" width="500" height="150" preserveAspectRatio="xMidYMid slice" xlink:href="../assets/morph_heterogeneous.png"/>')
    p.append(arrow(575,142,675,142)); p.append(roundbox(690,67,265,150,"warn","unconstrained-agent")); p.append(multiline(822,105,["Unconstrained","AI agent"],"body","middle",30)); p.append('<circle cx="822" cy="174" r="24" fill="#fff" stroke="#cf6540" stroke-width="3"/><path d="M804 174 H840 M822 156 V192" stroke="#cf6540" stroke-width="3"/>')
    p.append(arrow(970,142,1060,142)); p.append(mini_mask(1080,72,False)); p.append(mini_mask(1220,72,False)); p.append(mini_mask(1360,72,False)); p.append(text(1610,125,"Inconsistent masks", "body","middle")); p.append(text(1610,163,"and non-auditable metrics", "small","middle")); p.append('</g>')
    # B data registry
    p.append(panel(16,270,590,350,"B","Context-preserving dataset registry","B"))
    p.append(roundbox(50,335,210,220,"node","capture")); p.append(text(155,370,"Imaging capture","small-bold","middle")); p.append('<rect x="82" y="405" width="145" height="100" fill="#d9e1e4" stroke="#4f899b" stroke-width="2"/><circle cx="135" cy="452" r="32" fill="#8da0a5"/><circle cx="185" cy="448" r="27" fill="#a5b3b7"/>'); p.append(text(155,535,"bright-field images","tiny","middle"))
    p.append(arrow(270,445,315,445)); p.append(roundbox(325,335,245,220,"soft-node","metadata")); p.append(text(447,370,"Metadata","small-bold","middle")); p.append(multiline(447,410,["experiment design","date and condition","replicate","magnification","modality"],"small","middle",28)); p.append(text(310,592,"AAPS registers source paths, experimental context, and provenance.","tiny","middle")); p.append('</g>')
    # C routing
    p.append(panel(625,270,1259,350,"C","Image-prior-guided method routing and segmentation QC","C"))
    p.append(roundbox(660,335,280,220,"node","priors")); p.append(text(800,370,"Image priors","small-bold","middle")); p.append(multiline(800,410,["modality • quality","magnification • size","count • density","spatial distribution"],"small","middle",30))
    p.append(arrow(950,445,1020,445)); p.append('<circle cx="1080" cy="445" r="58" fill="#e9f3f6" stroke="#4f899b" stroke-width="3"/>'); p.append(text(1080,437,"AI", "body","middle")); p.append(text(1080,468,"agent", "small","middle")); p.append(arrow(1140,445,1200,445)); p.append(roundbox(1210,385,170,120,"soft-node","routing")); p.append(multiline(1295,425,["Method","routing"],"body","middle",30))
    methods=[("Cellpose","deep learning"),("Thresholding","intensity"),("Multiscale","segmentation"),("Annotation","image-assisted")]
    for i,(name,sub) in enumerate(methods):
        yy=315+i*69; p.append(roundbox(1415,yy,210,58,"node",f"method-{i}")); p.append(text(1520,yy+26,name,"small-bold","middle")); p.append(text(1520,yy+48,sub,"tiny","middle")); p.append(f'<path d="M1380 445 H1395 V{yy+29} H1405" class="dash"/>')
    p.append(arrow(1635,445,1690,445)); p.append(mini_mask(1705,398,True)); p.append(text(1765,535,"Segmentation QC","small-bold","middle")); p.append('</g>')
    # D metrics
    p.append(panel(16,640,925,600,"D","Experiment-aligned metric design","D"))
    p.append(roundbox(55,725,220,300,"node","experiment-design")); p.append(text(165,770,"Experiment design","small-bold","middle")); p.append(multiline(165,815,["conditions","treatments","replicates","time points"],"small","middle",38)); p.append(arrow(285,875,355,875))
    p.append(roundbox(365,705,535,470,"soft-node","metric-design")); p.append(text(632,750,"Selected quantitative readouts","body","middle"));
    p.append(metric_card(395,790,"Growth","total area","growth")); p.append(metric_card(565,790,"Differentiation","darkness P90","dark")); p.append(metric_card(735,790,"Shape","area / perimeter","shape")); p.append(metric_card(480,975,"Fusion","fusion index","fusion")); p.append(metric_card(650,975,"Centrality","edge distribution","shape")); p.append('</g>')
    # E outputs
    p.append(panel(960,640,924,600,"E","Auditable outputs and biological interpretation","E"))
    x0=995
    p.append(roundbox(x0,725,190,390,"node","database")); p.append(text(x0+95,765,"Results database","small-bold","middle")); p.append(multiline(x0+95,810,["per-image metrics","per-organoid metrics","source path","method + parameters","QC status"],"small","middle",37));
    p.append(arrow(1195,920,1240,920)); p.append(roundbox(1250,725,190,390,"node","representative")); p.append(text(1345,765,"Representative","small-bold","middle")); p.append(text(1345,795,"outputs","small-bold","middle")); p.append('<image x="1270" y="825" width="150" height="235" preserveAspectRatio="xMidYMid slice" xlink:href="../assets/morph_outputs.png"/>')
    p.append(arrow(1450,920,1495,920)); p.append(roundbox(1505,725,155,390,"node","plots")); p.append(text(1582,765,"Plots","small-bold","middle"));
    for j in range(3):
        yy=810+j*90; p.append(f'<rect x="1525" y="{yy}" width="115" height="70" fill="#fff" stroke="#9bbdca" stroke-width="2"/><path d="M1538 {yy+55} C1560 {yy+45} 1575 {yy+15} 1628 {yy+30}" fill="none" stroke="#4f899b" stroke-width="3"/>')
    p.append(arrow(1670,920,1710,920)); p.append(roundbox(1720,725,130,390,"soft-node","conclusion")); p.append(text(1785,765,"Report","small-bold","middle")); p.append(multiline(1785,820,["growth","differentiation","fusion","effect size","uncertainty"],"small","middle",45)); p.append('</g>')
    p.append('</svg>'); return '\n'.join(p)


if __name__ == "__main__":
    OUT.write_text(build(),encoding="utf-8"); print(OUT)
