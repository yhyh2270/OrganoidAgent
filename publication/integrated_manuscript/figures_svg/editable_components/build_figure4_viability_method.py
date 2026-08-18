"""Build the fluorescence-label and bright-field viability model method figure."""

from pathlib import Path

from PIL import Image

from build_figure2_aaps_editable import arrow, chart_icon, multiline, panel, roundbox, text


HERE=Path(__file__).resolve().parent; ASSETS=HERE.parent/"assets"
OUT=HERE/"figure4_viability_label_model_workflow.svg"; W,H=1900,1260


def prepare_assets():
    im=Image.open(ASSETS/"viability_label_reference.png").convert("RGB")
    im.crop((70,690,1110,1065)).save(ASSETS/"viability_representative_grid.png",quality=95)


def header():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    <title>Fluorescence-derived label construction and bright-field viability regression</title>
    <defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#496f7d"/></marker>
    <style>text{{font-family:Arial,Helvetica,sans-serif;fill:#17252b}} .panel{{fill:#f7fafb;stroke:#aac3cc;stroke-width:3}} .panel-label{{font-size:40px;font-weight:700}} .panel-title{{font-size:33px;font-weight:600}} .body{{font-size:27px}} .small{{font-size:23px}} .small-bold{{font-size:23px;font-weight:700}} .tiny{{font-size:19px}} .tiny-bold{{font-size:19px;font-weight:700}}
    .node{{fill:#fff;stroke:#4f899b;stroke-width:3}} .soft-node{{fill:#e9f3f6;stroke:#4f899b;stroke-width:3}} .arrow{{fill:none;stroke:#496f7d;stroke-width:3;marker-end:url(#arrowhead)}} .line{{fill:none;stroke:#496f7d;stroke-width:3}} .dash{{fill:none;stroke:#718f9a;stroke-width:2.5;stroke-dasharray:8 7;marker-end:url(#arrowhead)}}
    .green{{fill:#e7f5ec;stroke:#56a776;stroke-width:3}} .red{{fill:#fff0f0;stroke:#d66a6a;stroke-width:3}} .blue{{fill:#e8f1fb;stroke:#5d8fbd;stroke-width:3}} .stage{{fill:#dfe9f5;stroke:#496f7d;stroke-width:3}} .projection{{fill:#f5e3a3;stroke:#816b2b;stroke-width:3}} .head{{fill:#e7f1df;stroke:#496f7d;stroke-width:3}}
    .chart-bg{{fill:#fff;stroke:#7da5b2;stroke-width:2}} .chart-axis{{fill:none;stroke:#6c8790;stroke-width:2}}</style></defs><rect width="100%" height="100%" fill="white"/>'''


def channel_card(x,y,label,color):
    return f'<g><rect x="{x}" y="{y}" width="115" height="100" rx="8" fill="#101418" stroke="#4f899b" stroke-width="2"/><circle cx="{x+57}" cy="{y+50}" r="31" fill="{color}" opacity=".9"/><text x="{x+57}" y="{y+128}" class="small-bold" text-anchor="middle">{label}</text></g>'


def architecture_row(p,y,name,channels):
    p.append(roundbox(1150,y,165,70,"stage",f"stage-{name}")); p.append(text(1232,y+33,name,"body","middle")); p.append(text(1232,y+60,f"C={channels}","tiny","middle")); p.append(arrow(1320,y+35,1370,y+35)); p.append(roundbox(1380,y,185,70,"node",f"gap-{name}")); p.append(multiline(1472,y+28,["Global","avg pooling"],"small","middle",23)); p.append(arrow(1570,y+35,1615,y+35)); p.append(roundbox(1625,y,210,70,"projection",f"projection-{name}")); p.append(multiline(1730,y+28,["Linear","projection → 64"],"small","middle",23));


def build():
    prepare_assets(); p=[header()]
    # A QC
    p.append(panel(16,16,1225,430,"A","Fluorescence quality control and label construction","A"))
    p.append(channel_card(55,105,"LIVE","#2ec56f")); p.append(channel_card(195,105,"DEAD","#e44f5f")); p.append(arrow(325,155,405,155));
    p.append(roundbox(415,82,240,145,"node","references")); p.append(text(535,117,"Expert references","small-bold","middle")); p.append(multiline(535,151,["usable / unusable","few-shot examples"],"small","middle",26))
    p.append(arrow(665,155,735,155)); p.append(roundbox(745,82,205,145,"soft-node","rubric")); p.append(text(847,117,"QC rubric","small-bold","middle")); p.append(multiline(847,151,["signal","background","exposure","artifacts"],"tiny","middle",22))
    p.append(arrow(960,155,1030,155)); p.append(roundbox(1040,82,165,145,"soft-node","vision-qc")); p.append(text(1122,117,"Vision QC","small-bold","middle")); p.append(multiline(1122,153,["channel-wise","label + reason"],"tiny","middle",24))
    p.append(f'<path d="M1122 227 V265 H965" class="line"/>'); p.append(roundbox(735,260,215,80,"green","green-qc")); p.append(text(842,292,"Green-channel QC","small-bold","middle")); p.append(text(842,322,"usable / unusable","tiny","middle")); p.append(roundbox(970,260,215,80,"red","red-qc")); p.append(text(1077,292,"Red-channel QC","small-bold","middle")); p.append(text(1077,322,"usable / unusable","tiny","middle"));
    p.append(f'<path d="M735 300 H650 V360 H570" class="line"/><path d="M970 300 H925 V360 H570" class="line"/>'); p.append(roundbox(380,345,190,74,"node","sample-decision")); p.append(text(475,375,"Sample decision","small-bold","middle")); p.append(text(475,405,"both channels pass","tiny","middle")); p.append(arrow(370,382,275,382)); p.append(roundbox(55,345,210,74,"blue","retained")); p.append(text(160,375,"Retained samples","small-bold","middle")); p.append(text(160,405,"viability labels","tiny","middle")); p.append('</g>')
    # B stats
    p.append(panel(1260,16,624,430,"B","QC retention and label distribution","B"))
    vals=[("Green",87.1),("Red",88.0),("Sample",82.7)]
    bar_x=[1300,1435,1570]
    for i,(lab,val) in enumerate(vals):
        x=bar_x[i]; total=220; passed=total*val/100; failed=total-passed
        p.append(f'<rect x="{x}" y="{370-total}" width="65" height="{passed}" fill="#72a5b3"/><rect x="{x}" y="{370-total}" width="65" height="{failed}" fill="#c8d9de"/>'); p.append(text(x+32,128,f"{val:.1f}%","small-bold","middle")); p.append(text(x+32,408,lab,"small","middle"))
    p.append(text(1478,82,"QC retention","body","middle")); p.append(text(1785,105,"Viability distribution","small-bold","middle"));
    for i,hv in enumerate([12,18,25,38,54,76,105,132,158,180,150,92]):
        p.append(f'<rect x="1695" y="{370-hv}" width="14" height="{hv}" transform="translate({i*15} 0)" fill="#9fc7d8"/>')
    p.append(f'<path d="M1685 370 H1878 M1685 370 V150" class="line"/>'); p.append(text(1782,408,"viability score","small","middle")); p.append('</g>')
    # C representative paired data and deployment entry
    p.append(panel(16,465,670,775,"C","Training data and deployment","C"))
    p.append('<image x="45" y="535" width="610" height="245" preserveAspectRatio="xMidYMid meet" xlink:href="../assets/viability_representative_grid.png"/>'); p.append(text(350,815,"Representative bright-field, LIVE, DEAD, and ROI-mask pairs","small","middle"))
    p.append(roundbox(55,865,575,310,"soft-node","training-inference")); p.append(text(342,905,"Training","body","middle")); p.append(multiline(342,945,["bright-field ROI + fluorescence-derived label","→ regression and ranking objectives","→ optimized ConvNeXt checkpoint"],"small","middle",31)); p.append(f'<path d="M80 1035 H605" class="dash"/>'); p.append(text(342,1075,"Deployment inference","body","middle")); p.append(multiline(342,1112,["bright-field image → YOLO → SAM → ROI crop","→ checkpoint-1 → viability score"],"small","middle",31)); p.append(text(342,1198,"No fluorescence input is required during inference.","tiny-bold","middle")); p.append('</g>')
    # D architecture
    p.append(panel(705,465,1179,775,"D","ConvNeXt multi-scale viability regressor","D"))
    p.append(roundbox(745,735,170,150,"blue","roi-input")); p.append('<circle cx="830" cy="790" r="40" fill="#88999e"/><circle cx="855" cy="815" r="32" fill="#a9b6ba"/>'); p.append(text(830,850,"384 × 384 ROI","small-bold","middle")); p.append(text(830,878,"RGB bright-field","tiny","middle")); p.append(arrow(920,810,945,810)); p.append(f'<path d="M950 710 L1110 745 V875 L950 910Z" fill="#9fc4e5" stroke="#496f7d" stroke-width="3"/>'); p.append(multiline(1030,795,["ConvNeXt-Tiny","pretrained"],"body","middle",30));
    p.append(f'<path d="M1110 810 H1125 V595 H1140" class="line"/><path d="M1125 810 V775 H1140" class="line"/><path d="M1125 810 V955 H1140" class="line"/>')
    architecture_row(p,560,"C2",192); architecture_row(p,740,"C3",384); architecture_row(p,920,"C4",768)
    p.append(f'<path d="M1835 595 H1850 V1087 H1830" class="line"/><path d="M1835 775 H1850" class="line"/><path d="M1835 955 H1850" class="line"/>'); p.append(roundbox(1580,1045,250,85,"projection","fusion")); p.append(text(1705,1080,"Multi-scale fusion","small-bold","middle")); p.append(text(1705,1110,"concat[p2, p3, p4]","tiny","middle")); p.append(arrow(1570,1087,1515,1087)); p.append(roundbox(1265,1045,240,85,"head","mlp")); p.append(text(1385,1080,"Shared MLP + dropout","small-bold","middle")); p.append(text(1385,1110,"192 hidden units","tiny","middle")); p.append(arrow(1255,1087,1200,1087)); p.append(roundbox(990,1045,200,85,"head","head")); p.append(text(1090,1080,"Viability head","small-bold","middle")); p.append(text(1090,1110,"score ∈ [0,1]","tiny","middle")); p.append('</g>')
    p.append('</svg>'); return '\n'.join(p)


if __name__=="__main__": OUT.write_text(build(),encoding="utf-8"); print(OUT)
