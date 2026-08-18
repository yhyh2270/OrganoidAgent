"""Faithful editable reconstruction of the original five-panel AAPS figure."""

from pathlib import Path

from PIL import Image


HERE=Path(__file__).resolve().parent; ASSETS=HERE.parent/"assets"
SRC=ASSETS/"morphology_workflow_reference.png"
OUT=HERE/"figure3_original_faithful_editable.svg"
W,H=983,1591


def crop_assets():
    im=Image.open(SRC).convert("RGB")
    crops={
        "faithful_A_montage.png":(62,0,493,240),
        "faithful_A_examples.png":(746,0,978,225),
        "faithful_B_capture.png":(74,305,278,463),
        "faithful_C_qc.png":(685,500,962,825),
        "faithful_E_outputs.png":(300,1260,455,1575),
    }
    for name,box in crops.items(): im.crop(box).save(ASSETS/name,quality=95)


def tx(x,y,s,cls="body",anchor="start"):
    from xml.sax.saxutils import escape
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(s)}</text>'


def ml(x,y,lines,cls="body",anchor="start",dy=16):
    from xml.sax.saxutils import escape
    spans=''.join(f'<tspan x="{x}" dy="{0 if i==0 else dy}">{escape(v)}</tspan>' for i,v in enumerate(lines))
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{spans}</text>'


def box(x,y,w,h,cls="box",rx=10): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'
def arr(x1,y1,x2,y2,cls="arrow"): return f'<path d="M{x1} {y1} L{x2} {y2}" class="{cls}"/>'


def network(x,y):
    pts=[(0,24),(20,5),(42,18),(62,0),(80,25),(58,42),(32,38),(12,51)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,2),(2,5),(1,6),(3,6)]
    s=[f'<g transform="translate({x} {y})">']
    for a,b in edges:s.append(f'<line x1="{pts[a][0]}" y1="{pts[a][1]}" x2="{pts[b][0]}" y2="{pts[b][1]}" stroke="#173d55" stroke-width="1.5"/>')
    for px,py in pts:s.append(f'<circle cx="{px}" cy="{py}" r="4" fill="#173d55"/>')
    s.append('</g>'); return ''.join(s)


def metric(x,y,title,subtitle,kind,color):
    s=[f'<g id="metric-{kind}">{box(x,y,120,140,"metric")}',tx(x+60,y+22,title,"small-bold","middle"),tx(x+60,y+40,subtitle,"tiny","middle")]
    if kind=="growth":
        s.append(f'<circle cx="{x+60}" cy="{y+91}" r="35" fill="none" stroke="{color}" stroke-width="2"/>');
        for i in range(12): s.append(f'<circle cx="{x+35+(i%4)*17}" cy="{y+65+(i//4)*22}" r="4" fill="{color}"/>')
    elif kind=="dark":
        for i,hv in enumerate([15,24,37,52,63,50,35,22]): s.append(f'<rect x="{x+25+i*9}" y="{y+125-hv}" width="7" height="{hv}" fill="{color}"/>')
    elif kind in {"shape","edge"}:
        s.append(f'<path d="M{x+30} {y+115} C{x+18} {y+80} {x+42} {y+58} {x+68} {y+65} C{x+95} {y+48} {x+110} {y+83} {x+96} {y+111} C{x+72} {y+133} {x+45} {y+126} {x+30} {y+115}Z" fill="#f3f7f8" stroke="{color}" stroke-width="2"/>')
    elif kind=="fusion":
        s.append(f'<ellipse cx="{x+40}" cy="{y+95}" rx="23" ry="32" fill="#eef4f6" stroke="{color}" stroke-width="2"/><ellipse cx="{x+88}" cy="{y+95}" rx="23" ry="32" fill="#eef4f6" stroke="{color}" stroke-width="2"/>');s.append(arr(x+64,y+95,x+70,y+95))
    else:
        s.append(f'<circle cx="{x+60}" cy="{y+93}" r="37" fill="none" stroke="{color}" stroke-width="2"/><circle cx="{x+60}" cy="{y+93}" r="5" fill="{color}"/>');
        for a in range(0,360,45):
            import math
            s.append(f'<line x1="{x+60}" y1="{y+93}" x2="{x+60+35*math.cos(math.radians(a)):.1f}" y2="{y+93+35*math.sin(math.radians(a)):.1f}" stroke="{color}" stroke-width="1" stroke-dasharray="3 2"/>')
    s.append('</g>');return ''.join(s)


def build():
    crop_assets();p=[f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>Original AAPS morphology figure—faithful editable reconstruction</title><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#173d55"/></marker><style>text{{font-family:Arial,Helvetica,sans-serif;fill:#17324a}}.panel{{font-size:32px;font-weight:500;fill:#111}}.title{{font-size:18px;font-weight:700;fill:#145075}}.body{{font-size:15px}}.small{{font-size:13px}}.small-bold{{font-size:14px;font-weight:700}}.tiny{{font-size:11px}}.box{{fill:#fff;stroke:#6c8aa0;stroke-width:1.5}}.soft{{fill:#f5fbfd;stroke:#3992a8;stroke-width:1.7}}.warn{{fill:#fff;stroke:#df4b4b;stroke-width:1.7;stroke-dasharray:5 4}}.metric{{fill:#fff;stroke:#6c8aa0;stroke-width:1.4}}.arrow{{fill:none;stroke:#173d55;stroke-width:1.8;marker-end:url(#ah)}}.dash{{fill:none;stroke:#20809b;stroke-width:1.5;stroke-dasharray:3 3;marker-end:url(#ah)}}.ref{{display:none}}</style></defs><rect width="100%" height="100%" fill="white"/><g id="original-raster-reference" class="ref"><image width="983" height="1591" xlink:href="../assets/morphology_workflow_reference.png"/></g>''']
    # A
    p.append('<g id="panel-A">'+tx(5,32,"A","panel")+'<image x="65" y="4" width="425" height="232" preserveAspectRatio="xMidYMid slice" xlink:href="../assets/faithful_A_montage.png"/>')
    p.append(box(540,28,145,185,"warn"));p.append(ml(612,62,["Unconstrained","AI agent"],"title","middle",22));p.append(network(570,100));p.append(tx(612,248,"Heterogeneous organoid images","small-bold","middle"));
    p.append(arr(690,80,744,55,"dash"));p.append(arr(690,110,744,110,"dash"));p.append(arr(690,140,744,170,"dash"));p.append('<image x="748" y="0" width="230" height="220" preserveAspectRatio="xMidYMid meet" xlink:href="../assets/faithful_A_examples.png"/>');p.append('</g>')
    # B
    p.append('<g id="panel-B">'+tx(5,287,"B","panel")+box(72,287,205,177)+'<image x="88" y="307" width="170" height="143" preserveAspectRatio="xMidYMid meet" xlink:href="../assets/faithful_B_capture.png"/>')
    p.append(arr(280,375,298,375));p.append(box(296,263,470,207,"soft"));p.append(tx(531,281,"AAPS automatically organizes data and preserves context","title","middle"));p.append(box(304,289,155,165));p.append(tx(381,312,"Metadata","title","middle"));p.append(ml(381,340,["Experiment design","Date","Condition","Replicate","Magnification","Modality"],"tiny","middle",19));p.append(box(467,289,290,165));p.append(tx(612,312,"Dataset registry","title","middle"));
    # editable registry
    for yy in range(335,440,21):p.append(f'<line x1="535" y1="{yy}" x2="742" y2="{yy}" stroke="#8296a1" stroke-width=".8"/>')
    for xx in range(535,743,41):p.append(f'<line x1="{xx}" y1="325" x2="{xx}" y2="440" stroke="#8296a1" stroke-width=".8"/>')
    p.append(tx(638,328,"image ID   experiment   date   condition   replicate","tiny","middle"));p.append('</g>')
    # C
    p.append('<g id="panel-C">'+tx(5,508,"C","panel")+box(66,519,220,308)+tx(176,542,"Image priors (vision model)","title","middle")+network(132,570));p.append(ml(88,660,["✓ Modality","✓ Quality","✓ Magnification","✓ Size prior"],"small","start",43));p.append(ml(190,660,["✓ Count prior","✓ Density","✓ Distribution"],"small","start",43));p.append(arr(288,690,306,690,"dash"));p.append('<circle cx="350" cy="690" r="52" fill="#f5fbfd" stroke="#20809b" stroke-width="2"/>');p.append(network(311,665));p.append(tx(350,760,"AI agent","title","middle"));p.append(arr(404,690,418,690,"dash"));p.append(box(418,620,72,142,"soft"));p.append(ml(454,675,["Method","routing"],"title","middle",24));
    methods=[("Cellpose","deep learning"),("Thresholding","intensity"),("Multiscale","segmentation"),("Annotation","image-generation assisted")]
    for i,(a,b) in enumerate(methods):y=536+i*67;p.append(box(528,y,132,58,"box"));p.append(tx(594,y+24,a,"small-bold","middle"));p.append(tx(594,y+45,b,"tiny","middle"));p.append(f'<path d="M490 690 H510 V{y+29} H525" class="dash"/>')
    p.append('<image x="675" y="485" width="285" height="340" preserveAspectRatio="xMidYMid meet" xlink:href="../assets/faithful_C_qc.png"/>');p.append('</g>')
    # D
    p.append('<g id="panel-D">'+tx(5,870,"D","panel")+box(78,915,220,225)+tx(188,965,"Uses experiment design","title","middle")+ml(188,1060,["Conditions & factors","Treatments","Replicates & timepoints"],"small","middle",38));p.append(arr(305,1030,337,1030));p.append(box(337,843,423,372,"soft"));p.append(tx(548,875,"Metric design","title","middle"));p.append(tx(548,900,"Agent selects metrics aligned to goals and context","small","middle"));p.append(metric(352,913,"Growth","Total area","growth","#c68c34"));p.append(metric(485,913,"Differentiation","Darkness P90","dark","#dc5a61"));p.append(metric(618,913,"Area / Perimeter","Shape","shape","#557fc2"));p.append(metric(420,1065,"Edge structure","Edge roughness","edge","#dc7c4f"));p.append(metric(553,1065,"Fusion","Fusion index","fusion","#8754b7"));p.append(metric(686,1065,"Centrality","Mean to centroid","central","#e55590"));p.append('</g>')
    # E
    p.append('<g id="panel-E">'+tx(5,1260,"E","panel")+box(70,1230,208,345)+tx(174,1257,"Database (results)","title","middle")+tx(174,1280,"Per-organoid and per-image metrics","small","middle"));
    for yy in range(1310,1550,30):p.append(f'<line x1="80" y1="{yy}" x2="268" y2="{yy}" stroke="#8296a1" stroke-width=".8"/>')
    for xx in [80,126,172,218,268]:p.append(f'<line x1="{xx}" y1="1300" x2="{xx}" y2="1550" stroke="#8296a1" stroke-width=".8"/>')
    p.append(tx(174,1322,"image_id   organoid_id   area   P90   fusion","tiny","middle"));p.append(arr(281,1400,298,1400));p.append(box(300,1230,155,345));p.append(tx(377,1257,"Representative outputs","title","middle"));p.append('<image x="310" y="1280" width="135" height="270" preserveAspectRatio="xMidYMid slice" xlink:href="../assets/faithful_E_outputs.png"/>');p.append(arr(458,1400,482,1400));p.append(box(482,1230,205,345));p.append(tx(584,1257,"Plots","title","middle"));
    for i in range(3):y=1290+i*85;p.append(f'<rect x="495" y="{y}" width="180" height="72" fill="#fff" stroke="#6c8aa0"/><path d="M510 {y+55} C540 {y+35} 565 {y+15} 610 {y+40} C635 {y+55} 650 {y+25} 665 {y+30}" fill="none" stroke="#315d8a" stroke-width="2"/>')
    p.append(arr(690,1400,710,1400));p.append(box(710,1230,135,345));p.append(tx(777,1257,"Conclusions","title","middle"));p.append(ml(725,1320,["• growth response","• differentiation","• fusion events","• robust effects"],"small","start",55));p.append('</g></svg>');return '\n'.join(p)


if __name__=="__main__":OUT.write_text(build(),encoding="utf-8");print(OUT)
