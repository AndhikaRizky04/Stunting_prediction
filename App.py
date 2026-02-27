import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bisect
import io
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Stunting Balita",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0b1120; --navy-2: #111827; --navy-3: #1a2535; --navy-4: #1f2d42;
    --gold:   #c9a84c; --gold-lt:#e8d48b; --gold-dim:rgba(201,168,76,.15);
    --text-1: #f1f5f9; --text-2: #94a3b8; --text-3: #64748b;
    --border: rgba(255,255,255,.08); --border-gold:rgba(201,168,76,.3);
    --success:#34d399; --warn:#facc15; --orange:#fb923c; --danger:#f87171;
}
html,body,[class*="css"],.stApp,.main,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
section[data-testid="stSidebar"] {
    font-family:'Outfit',sans-serif!important;
    background-color:var(--navy)!important;
    color:var(--text-1)!important;
}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:1.5rem 2rem 4rem!important;}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{
    background:var(--navy-3)!important;
    border-right:1px solid var(--border-gold)!important;
}
section[data-testid="stSidebar"] * { color:var(--text-2)!important; }
section[data-testid="stSidebar"] .stRadio label { font-size:.9rem!important; }

/* ── HERO ── */
.hero-wrap{
    position:relative; text-align:center;
    padding:2.5rem 2rem 2.2rem; margin-bottom:2rem;
    border-radius:24px; background:var(--navy-3);
    border:1px solid var(--border-gold); overflow:hidden;
    animation:fadeDown .6s ease both;
}
.hero-wrap::before{
    content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse 80% 55% at 50% 0%,rgba(201,168,76,.12) 0%,transparent 70%);
}
.hero-wrap::after{
    content:''; position:absolute; bottom:-1px; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,var(--gold),transparent);
}
.hero-badge{
    display:inline-flex; align-items:center; gap:.4rem;
    background:var(--gold-dim); border:1px solid var(--border-gold);
    color:var(--gold-lt); font-size:.68rem; font-weight:600;
    letter-spacing:.14em; text-transform:uppercase;
    padding:.32rem .9rem; border-radius:100px; margin-bottom:1.2rem;
}
.hero-title{
    font-family:'Cormorant Garamond',serif!important;
    font-size:2.6rem!important; font-weight:300!important;
    letter-spacing:-.03em!important; line-height:1.1!important;
    color:var(--text-1)!important; margin:0 0 .3rem!important;
}
.hero-title em{font-style:italic;color:var(--gold-lt);}
.hero-divider{width:36px;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);margin:.9rem auto;}
.hero-sub{font-size:.85rem!important;color:var(--text-3)!important;font-weight:300!important;line-height:1.7!important;margin:0!important;}

/* ── SECTION LABEL ── */
.slabel{
    font-size:.63rem; font-weight:600; letter-spacing:.2em; text-transform:uppercase;
    color:var(--gold); margin-bottom:1rem; margin-top:.5rem;
    display:flex; align-items:center; gap:.7rem;
}
.slabel::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border-gold),transparent);}

/* ── GLASS CARD ── */
.gcard{
    background:var(--navy-3); border:1px solid var(--border);
    border-radius:18px; padding:1.5rem;
    transition:border-color .2s;
}
.gcard:hover{border-color:var(--border-gold);}

/* ── WIDGETS ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"]>div>div{
    background:var(--navy-3)!important; border:1.5px solid rgba(255,255,255,.1)!important;
    border-radius:12px!important; color:var(--text-1)!important;
    font-family:'Outfit',sans-serif!important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"]>div>div:focus-within{
    border-color:var(--gold)!important;
    box-shadow:0 0 0 3px var(--gold-dim)!important;
}
div[data-testid="stSelectbox"] ul{
    background:var(--navy-4)!important; border:1px solid var(--border-gold)!important;
    border-radius:12px!important; padding:.4rem!important;
}
div[data-testid="stSelectbox"] li{border-radius:8px!important;color:var(--text-2)!important;}
div[data-testid="stSelectbox"] li:hover{background:var(--gold-dim)!important;color:var(--gold-lt)!important;}
label[data-testid="stWidgetLabel"] p,div[data-testid="stWidgetLabel"] p{
    color:var(--text-3)!important; font-size:.8rem!important;
    font-weight:500!important; letter-spacing:.04em!important; text-transform:uppercase!important;
}
div[data-testid="stNumberInput"] button{
    background:var(--navy-4)!important; border:1px solid var(--border)!important;
    color:var(--text-3)!important; border-radius:8px!important;
}
div[data-testid="stSlider"] .stSlider>div>div>div{background:var(--gold)!important;}

/* ── CTA BUTTON ── */
.stButton>button{
    width:100%!important;
    background:linear-gradient(135deg,#9a6e1a,var(--gold) 50%,var(--gold-lt))!important;
    color:#0b1120!important; border:none!important; border-radius:14px!important;
    padding:.95rem 2rem!important; font-size:.92rem!important; font-weight:600!important;
    letter-spacing:.08em!important; text-transform:uppercase!important;
    font-family:'Outfit',sans-serif!important;
    box-shadow:0 8px 28px rgba(201,168,76,.3)!important;
    transition:all .3s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 14px 36px rgba(201,168,76,.4)!important;}

/* ── RESULT PANELS ── */
.rp{border-radius:22px;padding:2rem;text-align:center;position:relative;overflow:hidden;animation:resultPop .6s cubic-bezier(.34,1.56,.64,1) both;}
.rp::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 65% 45% at 50% 0%,var(--rp-glow) 0%,transparent 70%);}
.rp-normal{background:linear-gradient(160deg,#091a12,#0d2218);border:1px solid rgba(52,211,153,.25);--rp-glow:rgba(52,211,153,.12);}
.rp-waspada{background:linear-gradient(160deg,#1a1400,#221a00);border:1px solid rgba(250,204,21,.25);--rp-glow:rgba(250,204,21,.10);}
.rp-sedang{background:linear-gradient(160deg,#1a0e00,#221400);border:1px solid rgba(251,146,60,.25);--rp-glow:rgba(251,146,60,.11);}
.rp-tinggi{background:linear-gradient(160deg,#1a0909,#220d0d);border:1px solid rgba(248,113,113,.25);--rp-glow:rgba(248,113,113,.12);}
.rp-eyebrow{font-size:.63rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;opacity:.7;margin-bottom:.8rem;position:relative;}
.rp-icon{font-size:2.4rem;margin-bottom:.4rem;position:relative;line-height:1;}
.rp-headline{font-family:'Cormorant Garamond',serif;font-size:2.4rem;font-weight:600;letter-spacing:-.03em;line-height:1;margin-bottom:.4rem;position:relative;}
.rp-prob{font-size:.83rem;font-weight:400;opacity:.65;position:relative;}

/* ── Z-SCORE GRID ── */
.zsgrid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-bottom:1.1rem;}
.zscard{background:var(--navy-3);border:1px solid var(--border);border-radius:16px;padding:1.3rem;text-align:center;transition:border-color .2s,transform .2s;}
.zscard:hover{border-color:var(--border-gold);transform:translateY(-2px);}
.zstag{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;font-weight:600;color:var(--text-3);margin-bottom:.6rem;}
.zsval{font-family:'Cormorant Garamond',serif;font-size:2.6rem;font-weight:300;line-height:1;margin-bottom:.4rem;}
.zsdesc{font-size:.69rem;color:var(--text-3);margin-bottom:.5rem;}
.zsbadge{font-size:.71rem;font-weight:500;padding:.25rem .7rem;border-radius:100px;display:inline-block;}

/* ── GAUGE ── */
.gwrap{background:var(--navy-3);border:1px solid var(--border);border-radius:16px;padding:1.2rem 1.4rem;margin-bottom:1.1rem;}
.gtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.9rem;}
.gttl{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;font-weight:600;color:var(--text-3);}
.gnum{font-family:'Cormorant Garamond',serif;font-size:1.6rem;font-weight:300;line-height:1;}
.gtrack{height:6px;border-radius:100px;background:rgba(255,255,255,.07);position:relative;}
.gbar{position:absolute;left:0;top:0;height:100%;border-radius:100px;background:linear-gradient(90deg,#34d399 0%,#facc15 45%,#fb923c 70%,#f87171 100%);}
.gpip{position:absolute;top:50%;transform:translate(-50%,-50%);width:13px;height:13px;border-radius:50%;border:2px solid var(--navy);z-index:2;box-shadow:0 0 8px currentColor;}
.gtick{display:flex;justify-content:space-between;margin-top:.5rem;font-size:.66rem;color:var(--text-3);}

/* ── META PILLS ── */
.metarow{display:flex;gap:.8rem;margin-bottom:1.1rem;}
.mpill{flex:1;background:var(--navy-3);border:1px solid var(--border);border-radius:13px;padding:.9rem .7rem;text-align:center;transition:border-color .2s;}
.mpill:hover{border-color:var(--border-gold);}
.mplabel{font-size:.59rem;letter-spacing:.13em;text-transform:uppercase;color:var(--text-3);font-weight:600;}
.mpval{font-family:'Cormorant Garamond',serif;font-size:1.35rem;font-weight:300;color:var(--text-1);margin:.22rem 0 .12rem;line-height:1;}
.mpsub{font-size:.66rem;color:var(--text-3);}

/* ── INTERPRETATION BOX ── */
.ibox{background:var(--navy-3);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem;}
.ibox-title{font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:.8rem;}
.irow{display:flex;align-items:center;gap:.7rem;padding:.45rem .6rem;border-radius:8px;margin-bottom:.3rem;font-size:.86rem;}
.irow.active{border:1px solid;}
.irow-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}

/* ── RECOMMENDATION BOX ── */
.recbox{background:var(--navy-3);border:1px solid var(--border);border-left:3px solid;border-radius:14px;padding:1.1rem 1.3rem;font-size:.87rem;line-height:1.7;color:var(--text-2);margin-bottom:1rem;}
.recbox .acc{font-weight:500;}
.recbox .disc{font-size:.76rem;color:var(--text-3);margin-top:.6rem;font-style:italic;}
.rec-item{display:flex;align-items:flex-start;gap:.5rem;margin:.25rem 0;}
.rec-item::before{content:'→';color:var(--gold);flex-shrink:0;font-size:.9rem;}

/* ── CONFIDENCE BADGE ── */
.conf-badge{display:inline-flex;align-items:center;gap:.5rem;padding:.4rem 1rem;border-radius:100px;border:1px solid;font-size:.8rem;font-weight:500;margin-bottom:1rem;}

/* ── HISTORY TABLE ── */
.hist-wrap{overflow-x:auto;border-radius:14px;border:1px solid var(--border);}
.hist-wrap table{width:100%;border-collapse:collapse;font-size:.83rem;}
.hist-wrap th{background:var(--navy-4);color:var(--text-3);font-weight:500;padding:.7rem 1rem;text-align:left;letter-spacing:.05em;font-size:.72rem;text-transform:uppercase;}
.hist-wrap td{padding:.65rem 1rem;border-bottom:1px solid var(--border);color:var(--text-2);}
.hist-wrap tr:last-child td{border-bottom:none;}
.hist-wrap tr:hover td{background:rgba(255,255,255,.02);}

/* ── ABOUT PAGE ── */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;}
.stat-box{background:var(--navy-3);border:1px solid var(--border);border-radius:14px;padding:1.2rem;text-align:center;}
.stat-val{font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;color:var(--gold-lt);}
.stat-lbl{font-size:.7rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem;}

/* ── ALERTS ── */
div[data-testid="stAlert"]{background:rgba(201,168,76,.08)!important;border:1px solid var(--border-gold)!important;border-radius:12px!important;color:var(--gold-lt)!important;font-family:'Outfit',sans-serif!important;}
div[data-testid="stSpinner"] p{color:var(--text-2)!important;font-family:'Outfit',sans-serif!important;}

/* ── TABS ── */
div[data-testid="stTabs"] [role="tablist"]{gap:.3rem;border-bottom:1px solid var(--border-gold)!important;}
div[data-testid="stTabs"] [role="tab"]{
    background:transparent!important; border:none!important;
    color:var(--text-3)!important; font-family:'Outfit',sans-serif!important;
    font-size:.82rem!important; font-weight:500!important;
    padding:.55rem 1rem!important; border-radius:8px 8px 0 0!important;
    letter-spacing:.03em!important;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
    background:var(--gold-dim)!important; color:var(--gold-lt)!important;
    border-bottom:2px solid var(--gold)!important;
}
div[data-testid="stTabs"] [role="tab"]:hover{color:var(--text-1)!important;background:rgba(255,255,255,.04)!important;}

/* ── FOOTER ── */
.fnote{text-align:center;color:var(--text-3);font-size:.73rem;line-height:1.9;padding-top:1.5rem;margin-top:2.5rem;border-top:1px solid var(--border);}
.fnote span{color:var(--text-2);}

/* ── MATPLOTLIB DARK ── */
.stPlotlyChart,.element-container [data-testid="stImage"]{border-radius:14px;overflow:hidden;}

/* ── ANIMATIONS ── */
@keyframes fadeDown{from{opacity:0;transform:translateY(-16px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
@keyframes resultPop{from{opacity:0;transform:scale(.93);}to{opacity:1;transform:scale(1);}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WHO LMS TABLES (reused from earlier version)
# ─────────────────────────────────────────────
WFA_BOYS = [
    [0,0.3487,3.3464,0.14602],[1,0.2297,4.4709,0.13395],[2,0.1970,5.5675,0.12385],
    [3,0.1738,6.3762,0.11578],[4,0.1553,7.0023,0.10943],[5,0.1395,7.5105,0.10452],
    [6,0.1257,7.9340,0.10079],[7,0.1134,8.3030,0.09816],[8,0.1021,8.6420,0.09578],
    [9,0.0917,8.9496,0.09400],[10,0.0820,9.2422,0.09230],[11,0.0730,9.5238,0.09117],
    [12,0.0648,9.7870,0.09026],[15,0.0427,10.5244,0.08900],[18,0.0234,11.1945,0.08934],
    [24,-0.0108,12.4237,0.09258],[30,-0.0419,13.5638,0.09803],[36,-0.0705,14.6612,0.10456],
    [42,-0.0969,15.7243,0.11151],[48,-0.1213,16.7836,0.11856],[54,-0.1440,17.8740,0.12548],
    [60,-0.1650,19.0074,0.13208]
]
WFA_GIRLS = [
    [0,0.3809,3.2322,0.14171],[1,0.1714,4.1873,0.13724],[2,0.0962,5.1282,0.13000],
    [3,0.0569,5.8458,0.12516],[6,0.0150,7.2970,0.11556],[9,-0.0093,8.2003,0.11329],
    [12,-0.0315,8.9481,0.11382],[15,-0.0517,9.6792,0.11678],[18,-0.0702,10.3409,0.12012],
    [24,-0.1033,11.6031,0.12659],[30,-0.1323,12.8089,0.13326],[36,-0.1578,13.9348,0.13921],
    [42,-0.1802,15.0021,0.14441],[48,-0.1995,16.0586,0.14910],[54,-0.2163,17.1305,0.15331],
    [60,-0.2308,18.2298,0.15705]
]
WFL_BOYS = [
    [45.0,-0.3521,2.441,0.09182],[50.0,-0.3521,3.223,0.08221],[55.0,-0.3521,4.313,0.07654],
    [60.0,-0.3521,5.625,0.07495],[65.0,-0.3521,6.942,0.07651],[70.0,-0.3521,8.102,0.08058],
    [75.0,-0.3521,9.077,0.08648],[80.0,-0.3521,9.910,0.09376],[85.0,-0.3521,10.647,0.10163],
    [90.0,-0.3521,11.349,0.10933],[95.0,-0.3521,12.062,0.11649],[100.0,-0.3521,12.816,0.12325],
    [105.0,-0.3521,13.646,0.13012],[110.0,-0.3521,14.546,0.13741]
]
WFL_GIRLS = [
    [45.0,-0.3833,2.460,0.09029],[50.0,-0.3833,3.220,0.08007],[55.0,-0.3833,4.247,0.07228],
    [60.0,-0.3833,5.411,0.06818],[65.0,-0.3833,6.540,0.06720],[70.0,-0.3833,7.543,0.06840],
    [75.0,-0.3833,8.407,0.07120],[80.0,-0.3833,9.174,0.07548],[85.0,-0.3833,9.894,0.08111],
    [90.0,-0.3833,10.635,0.08798],[95.0,-0.3833,11.425,0.09601],[100.0,-0.3833,12.252,0.10485],
    [105.0,-0.3833,13.140,0.11447],[110.0,-0.3833,14.093,0.12490]
]

# WHO Height-for-Age median & SD (simplified, boys/girls)
# [age_months, -3SD, -2SD, median, +2SD, +3SD]
HFA_BOYS = [
    [0,43.6,46.1,49.9,53.7,56.2],[3,55.3,57.6,61.4,65.2,67.6],[6,61.7,64.2,67.6,71.6,74.0],
    [9,66.5,68.9,72.3,76.5,78.9],[12,70.5,73.1,75.7,80.5,83.0],[18,75.7,78.8,82.3,86.8,89.9],
    [24,80.0,83.3,87.1,91.9,95.2],[30,83.7,87.4,91.4,96.2,99.8],[36,86.4,90.2,94.9,101.4,103.6],
    [42,89.2,93.1,98.1,103.3,107.2],[48,92.5,96.3,101.5,107.6,111.4],[54,95.7,99.6,104.9,111.0,114.9],
    [60,98.7,102.8,108.2,114.3,118.2]
]
HFA_GIRLS = [
    [0,43.6,45.6,49.1,52.9,55.6],[3,53.5,55.6,59.8,64.0,66.1],[6,60.5,62.5,65.7,69.8,72.3],
    [9,64.6,67.1,70.1,74.7,77.2],[12,68.9,71.4,74.0,78.8,81.5],[18,74.4,77.2,80.7,86.0,88.8],
    [24,78.7,81.7,85.7,90.3,93.3],[30,82.9,86.2,90.3,95.0,98.4],[36,86.6,90.1,94.4,99.1,102.5],
    [42,90.2,94.0,98.4,103.5,107.0],[48,93.8,97.6,102.7,107.8,111.6],[54,97.2,101.1,106.4,112.0,115.8],
    [60,100.3,104.7,110.0,115.9,119.7]
]

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (from training notebook)
# ─────────────────────────────────────────────
FEATURE_IMPORTANCE = [
    ("Z-Score BB/TB (Wasting Index)",    18.4),
    ("Z-Score BB/U (Underweight Index)", 14.2),
    ("Tinggi / Panjang Badan",           11.7),
    ("Min Z-Score (BB/U & BB/TB)",        8.9),
    ("Umur × Z-Score Min",                7.3),
    ("Berat Badan",                       6.8),
    ("Skor Risiko Gizi Komposit",         5.9),
    ("Umur (Bulan)",                      5.4),
    ("Rasio BB/Tinggi",                   4.1),
    ("BMI Proxy",                         3.8),
    ("Prox ke -2SD (BB/TB)",              3.2),
    ("Prox ke -2SD (BB/U)",               2.8),
    ("Umur × BB/TB",                      2.5),
    ("Jenis Kelamin × Z-Score Min",       2.1),
    ("Window 1000 HPK",                   1.9),
]

MODEL_METRICS = {
    "Accuracy":  0.927,
    "Precision": 0.914,
    "Recall":    0.903,
    "F1-Score":  0.908,
    "AUC-ROC":   0.963,
}

# ─────────────────────────────────────────────
# WHO Z-SCORE FUNCTIONS
# ─────────────────────────────────────────────
def lms_zscore(X, L, M, S):
    if L == 0:
        z = np.log(X / M) / S
    else:
        z = ((X / M) ** L - 1) / (L * S)
    if z > 3:
        SD3pos  = M * (1 + L*S*3)**(1/L)
        SD23pos = SD3pos - M*(1+L*S*2)**(1/L)
        z = 3 + (X - SD3pos)/SD23pos
    elif z < -3:
        SD3neg  = M * (1+L*S*(-3))**(1/L)
        SD23neg = M*(1+L*S*(-2))**(1/L) - SD3neg
        z = -3 + (X - SD3neg)/SD23neg
    return round(z, 2)

def get_lms_age(age, sex):
    table = WFA_BOYS if sex=='L' else WFA_GIRLS
    ages  = [r[0] for r in table]
    a = max(0, min(60, int(round(age))))
    idx = bisect.bisect_left(ages, a)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0 and ages[idx] != a:
        a0,a1 = ages[idx-1], ages[idx]
        t = (a-a0)/(a1-a0) if a1!=a0 else 0
        return (table[idx-1][1]+t*(table[idx][1]-table[idx-1][1]),
                table[idx-1][2]+t*(table[idx][2]-table[idx-1][2]),
                table[idx-1][3]+t*(table[idx][3]-table[idx-1][3]))
    return table[idx][1], table[idx][2], table[idx][3]

def get_lms_height(h, sex):
    table = WFL_BOYS if sex=='L' else WFL_GIRLS
    hs = [r[0] for r in table]
    h  = max(hs[0], min(hs[-1], h))
    idx = bisect.bisect_left(hs, h)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0:
        h0,h1 = hs[idx-1], hs[idx]
        t = (h-h0)/(h1-h0) if h1!=h0 else 0
        return (table[idx-1][1]+t*(table[idx][1]-table[idx-1][1]),
                table[idx-1][2]+t*(table[idx][2]-table[idx-1][2]),
                table[idx-1][3]+t*(table[idx][3]-table[idx-1][3]))
    return table[idx][1], table[idx][2], table[idx][3]

def zscore_bbu(weight, age, sex):
    L,M,S = get_lms_age(age, sex)
    return lms_zscore(weight, L, M, S)

def zscore_bbtb(weight, h, sex, cara, age):
    hh = h
    if cara=='Terlentang' and age>=24: hh = h - 0.7
    elif cara=='Berdiri' and age<24:   hh = h + 0.7
    L,M,S = get_lms_height(hh, sex)
    return lms_zscore(weight, L, M, S)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(umur_bulan, jk, berat, tinggi, cara_ukur):
    zs_bb_u  = zscore_bbu(berat, umur_bulan, jk)
    zs_bb_tb = zscore_bbtb(berat, tinggi, jk, cara_ukur, umur_bulan)
    jk_enc   = 1 if jk=='L' else 0
    cara_enc = 1 if cara_ukur=='Berdiri' else 0
    bins = [-1,6,11,23,36,60]
    kel = 0
    for i in range(len(bins)-1):
        if umur_bulan>bins[i] and umur_bulan<=bins[i+1]: kel=[0,1,2,3,4][i]
    f_window=1 if umur_bulan<=23 else 0; f_mpasi=1 if 6<=umur_bulan<=23 else 0
    f_baduta=1 if umur_bulan<=23 else 0; age_sq=umur_bulan**2; age_log=np.log1p(umur_bulan)
    f_bb_sk=1 if zs_bb_u<-3 else 0; f_bb_k=1 if -3<=zs_bb_u<-2 else 0
    f_bb_n=1 if -2<=zs_bb_u<=1 else 0; f_bb_rl=1 if zs_bb_u>1 else 0; f_uw=1 if zs_bb_u<-2 else 0
    f_gb=1 if zs_bb_tb<-3 else 0; f_gk=1 if -3<=zs_bb_tb<-2 else 0
    f_gbk=1 if -2<=zs_bb_tb<=1 else 0; f_rgl=1 if 1<zs_bb_tb<=2 else 0
    f_gl=1 if 2<zs_bb_tb<=3 else 0; f_ob=1 if zs_bb_tb>3 else 0; f_wst=1 if zs_bb_tb<-2 else 0
    f_bb_mild=1 if -2.5<=zs_bb_u<-2 else 0; f_bbtb_mild=1 if -2.5<=zs_bb_tb<-2 else 0
    prox_bbu2=max(0,zs_bb_u-(-2)); prox_bbtb2=max(0,zs_bb_tb-(-2))
    prox_bbu3=max(0,zs_bb_u-(-3)); prox_bbtb3=max(0,zs_bb_tb-(-3))
    f_double=1 if f_uw==1 and f_wst==1 else 0; f_dsev=1 if f_bb_sk==1 and f_gb==1 else 0
    f_any_sv=1 if f_bb_sk==1 or f_gb==1 else 0
    skor=f_bb_sk*2+f_bb_k+f_gb*2+f_gk; jml_idx=(1 if f_uw else 0)+(1 if f_wst else 0)
    avg_z=(zs_bb_u+zs_bb_tb)/2; min_z=min(zs_bb_u,zs_bb_tb); max_z=max(zs_bb_u,zs_bb_tb)
    gap=zs_bb_u-zs_bb_tb; abs_g=abs(gap); prod=zs_bb_u*zs_bb_tb
    harm=(2*zs_bb_u*zs_bb_tb/(zs_bb_u+zs_bb_tb)) if (zs_bb_u!=0 and zs_bb_tb!=0) else 0.0
    bbu_sq=zs_bb_u**2; bbtb_sq=zs_bb_tb**2; bbu_cb=zs_bb_u**3; bbtb_cb=zs_bb_tb**3
    umur_x_risiko=umur_bulan*skor; umur_x_minzs=umur_bulan*min_z
    umur_x_bbu=umur_bulan*zs_bb_u; umur_x_bbtb=umur_bulan*zs_bb_tb
    bad_x_wst=f_baduta*f_wst; mpa_x_uw=f_mpasi*f_uw
    jk_x_minzs=jk_enc*min_z; jk_x_bbu=jk_enc*zs_bb_u; jk_x_risiko=jk_enc*skor
    bmi_proxy=berat/(tinggi/100)**2; rasio_bb_tb=berat/tinggi
    rasio_tb_um=tinggi/umur_bulan if umur_bulan>0 else 0; rasio_bb_um=berat/umur_bulan if umur_bulan>0 else 0
    rank_bbu=float(1/(1+np.exp(-zs_bb_u))); rank_bbtb=float(1/(1+np.exp(-zs_bb_tb)))
    row = {
        'umur_bulan':umur_bulan,'jk_encoded':jk_enc,'cara_ukur_encoded':cara_enc,
        'kel_usia_permenkes':float(kel),'f_window_1000hpk':f_window,
        'f_masa_mpasi':f_mpasi,'f_baduta':f_baduta,'age_sq':age_sq,'age_log':age_log,
        'zs_bb_u':zs_bb_u,'f_bb_sangat_kurang':f_bb_sk,'f_bb_kurang':f_bb_k,
        'f_bb_normal':f_bb_n,'f_bb_risiko_lebih':f_bb_rl,'f_underweight':f_uw,
        'zs_bb_tb':zs_bb_tb,'f_gizi_buruk':f_gb,'f_gizi_kurang':f_gk,'f_gizi_baik':f_gbk,
        'f_risiko_gizi_lebih':f_rgl,'f_gizi_lebih':f_gl,'f_obesitas':f_ob,'f_wasting':f_wst,
        'f_bb_mild':f_bb_mild,'f_bbtb_mild':f_bbtb_mild,
        'prox_bbu_ke_minus2':prox_bbu2,'prox_bbtb_ke_minus2':prox_bbtb2,
        'prox_bbu_ke_minus3':prox_bbu3,'prox_bbtb_ke_minus3':prox_bbtb3,
        'f_double_malnutrisi':f_double,'f_double_severe':f_dsev,'f_any_severe':f_any_sv,
        'skor_risiko_gizi':skor,'jml_indeks_masalah':jml_idx,
        'avg_zs_bbu_bbtb':avg_z,'min_zs_bbu_bbtb':min_z,'max_zs_bbu_bbtb':max_z,
        'gap_bbu_bbtb':gap,'abs_gap_bbu_bbtb':abs_g,'product_zs':prod,'harmonic_zs':harm,
        'zs_bbu_sq':bbu_sq,'zs_bbtb_sq':bbtb_sq,'zs_bbu_cb':bbu_cb,'zs_bbtb_cb':bbtb_cb,
        'umur_x_risiko':umur_x_risiko,'umur_x_minzs':umur_x_minzs,
        'umur_x_bbu':umur_x_bbu,'umur_x_bbtb':umur_x_bbtb,
        'baduta_x_wasting':bad_x_wst,'mpasi_x_underw':mpa_x_uw,
        'jk_x_minzs':jk_x_minzs,'jk_x_bbu':jk_x_bbu,'jk_x_risiko':jk_x_risiko,
        'bmi_proxy':bmi_proxy,'rasio_bb_tb':rasio_bb_tb,
        'rasio_tb_umur':rasio_tb_um,'rasio_bb_umur':rasio_bb_um,
        'berat':berat,'tinggi':tinggi,'rank_bbu':rank_bbu,'rank_bbtb':rank_bbtb
    }
    features = [
        'umur_bulan','jk_encoded','cara_ukur_encoded','kel_usia_permenkes',
        'f_window_1000hpk','f_masa_mpasi','f_baduta','age_sq','age_log','zs_bb_u',
        'f_bb_sangat_kurang','f_bb_kurang','f_bb_normal','f_bb_risiko_lebih','f_underweight',
        'zs_bb_tb','f_gizi_buruk','f_gizi_kurang','f_gizi_baik','f_risiko_gizi_lebih',
        'f_gizi_lebih','f_obesitas','f_wasting','f_bb_mild','f_bbtb_mild',
        'prox_bbu_ke_minus2','prox_bbtb_ke_minus2','prox_bbu_ke_minus3','prox_bbtb_ke_minus3',
        'f_double_malnutrisi','f_double_severe','f_any_severe','skor_risiko_gizi',
        'jml_indeks_masalah','avg_zs_bbu_bbtb','min_zs_bbu_bbtb','max_zs_bbu_bbtb',
        'gap_bbu_bbtb','abs_gap_bbu_bbtb','product_zs','harmonic_zs',
        'zs_bbu_sq','zs_bbtb_sq','zs_bbu_cb','zs_bbtb_cb',
        'umur_x_risiko','umur_x_minzs','umur_x_bbu','umur_x_bbtb',
        'baduta_x_wasting','mpasi_x_underw','jk_x_minzs','jk_x_bbu','jk_x_risiko',
        'bmi_proxy','rasio_bb_tb','rasio_tb_umur','rasio_bb_umur','berat','tinggi',
        'rank_bbu','rank_bbtb'
    ]
    df = pd.DataFrame([row])[features]
    return df.replace([np.inf,-np.inf], np.nan).fillna(0), zs_bb_u, zs_bb_tb

# ─────────────────────────────────────────────
# RISK TIER HELPER
# ─────────────────────────────────────────────
def risk_tier(pct):
    if pct < 45:
        return dict(
            cls="rp-normal", icon="✓", title="Tumbuh Kembang Normal",
            color="#34d399", eyebrow="Status Pertumbuhan · Aman",
            level="Rendah", badge_bg="rgba(52,211,153,.15)", badge_col="#34d399",
            recs=[
                "Pertahankan asupan gizi seimbang dan ASI/MPASI sesuai usia",
                "Lakukan pemantauan rutin di posyandu setiap bulan",
                "Pastikan jadwal imunisasi lengkap dan terpenuhi",
                "Stimulasi tumbuh kembang dengan bermain aktif",
            ]
        )
    elif pct < 65:
        return dict(
            cls="rp-waspada", icon="○", title="Perlu Diwaspadai",
            color="#facc15", eyebrow="Status Pertumbuhan · Perlu Perhatian",
            level="Sedang-Rendah", badge_bg="rgba(250,204,21,.15)", badge_col="#facc15",
            recs=[
                "Konsultasikan ke petugas gizi atau bidan untuk evaluasi",
                "Pastikan kecukupan asupan protein, zat besi, dan zinc",
                "Pantau berat dan tinggi badan setiap bulan",
                "Perhatikan pola makan dan jadwal makan anak",
            ]
        )
    elif pct < 80:
        return dict(
            cls="rp-sedang", icon="⚡", title="Risiko Stunting Sedang",
            color="#fb923c", eyebrow="Perhatian · Intervensi Gizi Dianjurkan",
            level="Sedang-Tinggi", badge_bg="rgba(251,146,60,.15)", badge_col="#fb923c",
            recs=[
                "Segera konsultasikan ke dokter atau ahli gizi anak",
                "Evaluasi pola makan, asupan kalori dan protein harian",
                "Pertimbangkan suplemen gizi mikro (vitamin A, zinc, zat besi)",
                "Pemantauan pertumbuhan setiap 2 minggu",
            ]
        )
    else:
        return dict(
            cls="rp-tinggi", icon="⚠️", title="Risiko Stunting Tinggi",
            color="#f87171", eyebrow="Perhatian · Tindak Lanjut Segera",
            level="Tinggi", badge_bg="rgba(248,113,113,.15)", badge_col="#f87171",
            recs=[
                "Bawa ke puskesmas atau dokter spesialis anak segera",
                "Diperlukan intervensi gizi intensif dan terstruktur",
                "Evaluasi faktor penyebab: infeksi berulang, MPASI tidak adekuat",
                "Pemantauan pertumbuhan setiap minggu selama intervensi",
            ]
        )

def bbu_style(z):
    if z<-3: return "Sangat Kurang","#f87171","rgba(248,113,113,.14)"
    if z<-2: return "Kurang","#fb923c","rgba(251,146,60,.14)"
    if z<=1: return "Normal","#34d399","rgba(52,211,153,.14)"
    return "Risiko Lebih","#facc15","rgba(250,204,21,.14)"

def bbtb_style(z):
    if z<-3: return "Gizi Buruk","#f87171","rgba(248,113,113,.14)"
    if z<-2: return "Gizi Kurang","#fb923c","rgba(251,146,60,.14)"
    if z<=1: return "Gizi Baik","#34d399","rgba(52,211,153,.14)"
    if z<=2: return "Risiko Lebih","#facc15","rgba(250,204,21,.14)"
    if z<=3: return "Gizi Lebih","#f97316","rgba(249,115,22,.14)"
    return "Obesitas","#ef4444","rgba(239,68,68,.14)"

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)]*len(models)
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for model, w in zip(self.models, self.weights):
            probs += w * model.predict_proba(X)
        return probs

@st.cache_resource
def load_model():
    import sys, types
    mod = types.ModuleType("__main__")
    mod.EnsembleModel = EnsembleModel
    sys.modules["__main__"] = mod
    data = joblib.load("model.pkl")
    if isinstance(data, dict):
        model     = data["model"]
        threshold = float(data.get("threshold", 0.45))
        features  = data.get("features", None)
    else:
        model = data; threshold = 0.45; features = None
    if features is None:
        try: features = joblib.load("fitur_training.pkl")
        except: features = []
    return model, threshold, features

# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────
def validate_inputs(umur, berat, tinggi, jk):
    warnings = []
    # Berat realistis per usia
    berat_min = {0:2.0,6:5.5,12:7.0,24:9.0,36:11.0,48:13.0,60:15.0}
    berat_max = {0:5.5,6:10.5,12:14.0,24:17.0,36:20.0,48:23.0,60:27.0}
    for k in sorted(berat_min.keys()):
        if umur >= k:
            bmin, bmax = berat_min[k], berat_max[k]
    if berat < bmin:
        warnings.append(f"⚠️ Berat badan {berat} kg terlalu rendah untuk usia {umur} bulan (min ~{bmin} kg)")
    if berat > bmax:
        warnings.append(f"⚠️ Berat badan {berat} kg terlalu tinggi untuk usia {umur} bulan (maks ~{bmax} kg)")
    # Tinggi realistis per usia
    tinggi_min = {0:44,6:60,12:68,24:78,36:86,48:93,60:100}
    tinggi_max = {0:57,6:74,12:84,24:97,36:107,48:116,60:123}
    for k in sorted(tinggi_min.keys()):
        if umur >= k:
            tmin, tmax = tinggi_min[k], tinggi_max[k]
    if tinggi < tmin:
        warnings.append(f"⚠️ Tinggi/panjang {tinggi} cm tidak realistis untuk usia {umur} bulan (min ~{tmin} cm)")
    if tinggi > tmax:
        warnings.append(f"⚠️ Tinggi/panjang {tinggi} cm tidak realistis untuk usia {umur} bulan (maks ~{tmax} cm)")
    # BMI check
    bmi = berat / (tinggi/100)**2
    if bmi < 10: warnings.append("⚠️ BMI sangat rendah — periksa kembali data berat dan tinggi")
    if bmi > 30: warnings.append("⚠️ BMI sangat tinggi — periksa kembali data berat dan tinggi")
    return warnings

# ─────────────────────────────────────────────
# WHO GROWTH CHART
# ─────────────────────────────────────────────
def plot_growth_chart(umur, tinggi, berat, sex):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#1a2535")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#64748b", labelsize=9)
        ax.spines['bottom'].set_color("#1f2d42")
        ax.spines['left'].set_color("#1f2d42")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.8)

    hfa = HFA_BOYS if sex=="L" else HFA_GIRLS
    ages_hfa = [r[0] for r in hfa]
    s3neg = [r[1] for r in hfa]; s2neg = [r[2] for r in hfa]
    med   = [r[3] for r in hfa]; s2pos = [r[4] for r in hfa]; s3pos = [r[5] for r in hfa]

    ax = axes[0]
    ax.fill_between(ages_hfa, s3neg, s2neg, color="#f87171", alpha=0.15)
    ax.fill_between(ages_hfa, s2neg, s2pos, color="#34d399", alpha=0.10)
    ax.fill_between(ages_hfa, s2pos, s3pos, color="#fb923c", alpha=0.12)
    ax.plot(ages_hfa, s3neg, "--", color="#f87171", lw=1.2, alpha=.7, label="-3 SD")
    ax.plot(ages_hfa, s2neg, "--", color="#facc15", lw=1.2, alpha=.7, label="-2 SD")
    ax.plot(ages_hfa, med,   "-",  color="#64748b", lw=1.5, alpha=.8, label="Median")
    ax.plot(ages_hfa, s2pos, "--", color="#facc15", lw=1.2, alpha=.7, label="+2 SD")
    ax.plot(ages_hfa, s3pos, "--", color="#f87171", lw=1.2, alpha=.7, label="+3 SD")
    ax.scatter([umur], [tinggi], color="#c9a84c", s=90, zorder=5, label="Anak ini")
    ax.axvline(x=umur, color="#c9a84c", lw=0.8, linestyle=":", alpha=.6)
    ax.axhline(y=tinggi, color="#c9a84c", lw=0.8, linestyle=":", alpha=.6)
    ax.set_xlabel("Umur (Bulan)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Tinggi/Panjang Badan (cm)", color="#94a3b8", fontsize=9)
    ax.set_title(f"Tinggi Badan / Umur  ·  {'Laki-laki' if sex=='L' else 'Perempuan'}",
                 color="#e8d48b", fontsize=10, fontweight='normal', pad=10)
    ax.legend(fontsize=8, facecolor="#1a2535", edgecolor="#1f2d42",
              labelcolor="#94a3b8", loc="upper left")

    # Weight-for-age
    wfa = WFA_BOYS if sex=="L" else WFA_GIRLS
    ages_w = [r[0] for r in wfa]
    meds_w = [r[2] for r in wfa]
    # approximate -2SD and +2SD from LMS
    s2n_w = []; s2p_w = []; s3n_w = []; s3p_w = []
    for row in wfa:
        L,M,S = row[1],row[2],row[3]
        s2n_w.append(M*(1+L*S*(-2))**(1/L) if L!=0 else M*np.exp(-2*S))
        s2p_w.append(M*(1+L*S*2)**(1/L)   if L!=0 else M*np.exp(2*S))
        s3n_w.append(M*(1+L*S*(-3))**(1/L) if L!=0 else M*np.exp(-3*S))
        s3p_w.append(M*(1+L*S*3)**(1/L)   if L!=0 else M*np.exp(3*S))

    ax2 = axes[1]
    ax2.fill_between(ages_w, s3n_w, s2n_w, color="#f87171", alpha=0.15)
    ax2.fill_between(ages_w, s2n_w, s2p_w, color="#34d399", alpha=0.10)
    ax2.fill_between(ages_w, s2p_w, s3p_w, color="#fb923c", alpha=0.12)
    ax2.plot(ages_w, s3n_w, "--", color="#f87171", lw=1.2, alpha=.7, label="-3 SD")
    ax2.plot(ages_w, s2n_w, "--", color="#facc15", lw=1.2, alpha=.7, label="-2 SD")
    ax2.plot(ages_w, meds_w, "-", color="#64748b", lw=1.5, alpha=.8, label="Median")
    ax2.plot(ages_w, s2p_w, "--", color="#facc15", lw=1.2, alpha=.7, label="+2 SD")
    ax2.plot(ages_w, s3p_w, "--", color="#f87171", lw=1.2, alpha=.7, label="+3 SD")
    ax2.scatter([umur], [berat], color="#c9a84c", s=90, zorder=5, label="Anak ini")
    ax2.axvline(x=umur, color="#c9a84c", lw=0.8, linestyle=":", alpha=.6)
    ax2.axhline(y=berat, color="#c9a84c", lw=0.8, linestyle=":", alpha=.6)
    ax2.set_xlabel("Umur (Bulan)", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Berat Badan (kg)", color="#94a3b8", fontsize=9)
    ax2.set_title(f"Berat Badan / Umur  ·  {'Laki-laki' if sex=='L' else 'Perempuan'}",
                  color="#e8d48b", fontsize=10, fontweight='normal', pad=10)
    ax2.legend(fontsize=8, facecolor="#1a2535", edgecolor="#1f2d42",
               labelcolor="#94a3b8", loc="upper left")

    # legend patch for zones
    p1 = mpatches.Patch(facecolor="#34d399", alpha=.3, label="Normal (−2 s.d. +2 SD)")
    p2 = mpatches.Patch(facecolor="#facc15", alpha=.3, label="Borderline (−3 s.d. −2 SD)")
    p3 = mpatches.Patch(facecolor="#f87171", alpha=.3, label="Kritis (< −3 SD)")
    for ax in axes:
        ax.legend(handles=[p1,p2,p3,
            plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#c9a84c',markersize=7,label='Posisi Anak')],
            fontsize=7.5, facecolor="#1a2535", edgecolor="#1f2d42",
            labelcolor="#94a3b8", loc="upper left")

    plt.tight_layout(pad=2.5)
    return fig

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────
def plot_feature_importance():
    feats = FEATURE_IMPORTANCE[:12]
    names = [f[0] for f in feats]
    vals  = [f[1] for f in feats]
    colors = []
    for v in vals:
        if v >= 12:   colors.append("#c9a84c")
        elif v >= 7:  colors.append("#4fd1c5")
        elif v >= 4:  colors.append("#94a3b8")
        else:         colors.append("#475569")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#1a2535")
    ax.set_facecolor("#111827")

    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1],
                   height=0.62, edgecolor="none")
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va='center', ha='left',
                color="#94a3b8", fontsize=8.5)

    ax.set_xlabel("Kontribusi (%)", color="#64748b", fontsize=9)
    ax.set_title("Feature Importance — Faktor Penentu Prediksi",
                 color="#e8d48b", fontsize=11, fontweight='normal', pad=12)
    ax.tick_params(colors="#64748b", labelsize=8.5)
    ax.spines['bottom'].set_color("#1f2d42")
    ax.spines['left'].set_color("#1f2d42")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(vals)+4)
    ax.grid(axis='x', color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.8)

    plt.tight_layout(pad=2)
    return fig

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1.5rem;'>
        <div style='font-size:2rem;margin-bottom:.4rem;'>🌿</div>
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.2rem;color:#e8d48b;font-weight:300;'>
            Prediksi Stunting
        </div>
        <div style='font-size:.7rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem;'>
            Balita · AI System
        </div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(201,168,76,.2);margin-bottom:1.2rem;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Prediksi Tunggal",
         "📊  Grafik Pertumbuhan",
         "🧬  Feature Importance",
         "⚡  Simulasi Risiko",
         "📋  Riwayat Prediksi",
         "📁  Prediksi Batch (Excel)",
         "📚  Tentang Model"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:1.5rem 0 .8rem;'>
    <div style='font-size:.7rem;color:#334155;text-align:center;line-height:1.7;'>
        Permenkes No. 2 / 2020<br>WHO 2006 Growth Standards<br>
        Ensemble CatBoost + XGBoost
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO (always shown)
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🌿 &nbsp; Sistem Skrining AI · Permenkes No. 2 Tahun 2020</div>
    <h1 class="hero-title">Prediksi <em>Stunting</em> Balita</h1>
    <div class="hero-divider"></div>
    <p class="hero-sub">
        Analisis antropometri berbasis WHO 2006 Growth Standards
        &nbsp;·&nbsp; Ensemble CatBoost + XGBoost &nbsp;·&nbsp; Feature Engineering 70+ Variabel
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PREDIKSI TUNGGAL
# ═══════════════════════════════════════════════════════════════
if menu.startswith("🏠"):
    st.markdown('<p class="slabel">Data Antropometri Anak</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        umur_bulan   = st.number_input("Umur (Bulan)", 0, 60, 24, 1)
        berat        = st.number_input("Berat Badan (kg)", 1.0, 35.0, 12.0, 0.1, format="%.1f")
    with col2:
        tinggi       = st.number_input("Tinggi / Panjang Badan (cm)", 40.0, 130.0, 87.0, 0.1, format="%.1f")
        jenis_kelamin= st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"])
    with col3:
        cara_ukur    = st.selectbox("Cara Pengukuran",
                                    ["Terlentang — Panjang Badan", "Berdiri — Tinggi Badan"],
                                    help="Terlentang < 24 bln · Berdiri ≥ 24 bln")
        nama_anak    = st.text_input("Nama Anak (opsional)", placeholder="contoh: Budi")

    # Validation warnings
    cu_mode = "Berdiri" if cara_ukur.startswith("Berdiri") else "Terlentang"
    jk = "L" if jenis_kelamin=="Laki-laki" else "P"

    if umur_bulan < 24 and cu_mode=="Berdiri":
        st.warning("Untuk usia < 24 bulan, disarankan pengukuran Terlentang (panjang badan).")
    elif umur_bulan >= 24 and cu_mode=="Terlentang":
        st.warning("Untuk usia ≥ 24 bulan, disarankan pengukuran Berdiri (tinggi badan).")

    val_warns = validate_inputs(umur_bulan, berat, tinggi, jk)
    for w in val_warns:
        st.warning(w)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Analisis & Prediksi Sekarang", key="predict_main")

    if predict_btn:
        with st.spinner("Menganalisis data antropometri..."):
            try:
                model, threshold, _ = load_model()
                X_input, zs_bbu, zs_bbtb = build_features(umur_bulan, jk, berat, tinggi, cu_mode)
                proba         = model.predict_proba(X_input)[0]
                prob_stunting = proba[1]
                pct           = prob_stunting * 100
                confidence    = max(proba) * 100
            except Exception as e:
                st.error(f"Error saat prediksi: {e}"); st.stop()

        tier = risk_tier(pct)
        st.session_state.last_result = {
            "umur": umur_bulan, "berat": berat, "tinggi": tinggi,
            "jk": jenis_kelamin, "nama": nama_anak,
            "pct": pct, "zs_bbu": zs_bbu, "zs_bbtb": zs_bbtb,
            "tier": tier, "confidence": confidence, "cara": cu_mode
        }

        # Save to history
        st.session_state.history.append({
            "Waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Nama": nama_anak or "—",
            "Umur (bln)": umur_bulan,
            "BB (kg)": berat,
            "TB (cm)": tinggi,
            "JK": jenis_kelamin,
            "Prob (%)": round(pct, 1),
            "Hasil": tier["title"],
        })

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Hasil Analisis</p>', unsafe_allow_html=True)

        # ── RESULT PANEL
        rp = tier
        bar_pct = max(2, min(97, pct))
        st.markdown(f"""
        <div class="rp {rp['cls']}">
            <div class="rp-eyebrow" style="color:{rp['color']};">{rp['eyebrow']}</div>
            <div class="rp-icon">{rp['icon']}</div>
            <div class="rp-headline" style="color:{rp['color']};">{rp['title']}</div>
            <div class="rp-prob">Probabilitas stunting &nbsp;—&nbsp;
                <strong style="color:{rp['color']};font-size:1rem;">{pct:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE SCORE
        conf_col = "#34d399" if confidence >= 80 else ("#facc15" if confidence >= 60 else "#fb923c")
        conf_lbl = "Tinggi" if confidence >= 80 else ("Sedang" if confidence >= 60 else "Rendah")
        st.markdown(f"""
        <div style="text-align:center;margin:.8rem 0 1.2rem;">
            <span class="conf-badge"
                style="background:rgba(201,168,76,.1);border-color:var(--border-gold);color:var(--gold-lt);">
                🧠 Kepercayaan Model:
                <strong style="color:{conf_col};">{confidence:.1f}%</strong>
                &nbsp;<span style="color:{conf_col};font-size:.75rem;">(Keyakinan {conf_lbl})</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        # ── Z-SCORE CARDS
        bc,bcol,bbg = bbu_style(zs_bbu)
        btbc,btbcol,btbbg = bbtb_style(zs_bbtb)
        with r1:
            st.markdown(f"""
            <div class="zsgrid">
                <div class="zscard">
                    <div class="zstag">Z-Score BB / Umur</div>
                    <div class="zsval" style="color:{bcol};">{zs_bbu:+.2f}</div>
                    <div class="zsdesc">Berat Badan menurut Umur</div>
                    <span class="zsbadge" style="background:{bbg};color:{bcol};">{bc}</span>
                </div>
                <div class="zscard">
                    <div class="zstag">Z-Score BB / Tinggi</div>
                    <div class="zsval" style="color:{btbcol};">{zs_bbtb:+.2f}</div>
                    <div class="zsdesc">Berat Badan menurut Tinggi</div>
                    <span class="zsbadge" style="background:{btbbg};color:{btbcol};">{btbc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── GAUGE
            st.markdown(f"""
            <div class="gwrap">
                <div class="gtop">
                    <span class="gttl">Indeks Probabilitas Stunting</span>
                    <span class="gnum" style="color:{rp['color']};">{pct:.1f}<span style="font-size:.9rem;opacity:.55;">%</span></span>
                </div>
                <div class="gtrack">
                    <div class="gbar" style="width:100%;"></div>
                    <div class="gpip" style="left:{bar_pct}%;color:{rp['color']};background:{rp['color']};"></div>
                </div>
                <div class="gtick"><span>Normal</span><span>Batas 45%</span><span>Stunting</span></div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # ── INTERPRETASI HASIL
            bands = [
                ("<b>0 – 44%</b>", "Normal / Aman",       "#34d399", pct < 45),
                ("<b>45 – 64%</b>","Perlu Diwaspadai",    "#facc15", 45<=pct<65),
                ("<b>65 – 79%</b>","Risiko Stunting Sedang","#fb923c",65<=pct<80),
                ("<b>≥ 80%</b>",   "Risiko Stunting Tinggi","#f87171",pct>=80),
            ]
            bands_html = ""
            for rng, lbl, col, is_active in bands:
                active_style = f"background:{col}18;border-color:{col}40;" if is_active else "border-color:transparent;"
                arrow = f"<span style='color:{col};font-size:1rem;'>◀</span>" if is_active else ""
                bands_html += f"""
                <div class="irow {'active' if is_active else ''}"
                     style="{active_style}">
                    <div class="irow-dot" style="background:{col};{'box-shadow:0 0 6px '+col if is_active else ''}"></div>
                    <span style="color:{'#f1f5f9' if is_active else '#64748b'};flex:1;">{rng} — {lbl}</span>
                    {arrow}
                </div>"""

            st.markdown(f"""
            <div class="ibox">
                <div class="ibox-title">📊 Interpretasi Hasil Prediksi</div>
                <div style="font-size:.8rem;color:#94a3b8;margin-bottom:.8rem;">
                    Probabilitas anak: <strong style="color:{rp['color']};">{pct:.1f}%</strong>
                    &nbsp;·&nbsp; Level: <strong style="color:{rp['color']};">{rp['level']}</strong>
                </div>
                {bands_html}
                <div style="font-size:.72rem;color:#475569;margin-top:.8rem;font-style:italic;">
                    Ambang batas ditentukan berdasarkan optimasi F1-Score model.
                    Bukan pengganti diagnosis medis.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── META PILLS
        bmi_val = berat/(tinggi/100)**2
        kel_map = {0:"ASI Eksklusif",1:"MPASI Awal",2:"Baduta / 1000 HPK",3:"Batita",4:"Balita"}
        bins_k  = [-1,6,11,23,36,60]; kel = 0
        for i in range(len(bins_k)-1):
            if umur_bulan>bins_k[i] and umur_bulan<=bins_k[i+1]: kel=[0,1,2,3,4][i]
        hpk_col  = "#34d399" if umur_bulan<=23 else "#64748b"
        hpk_text = "Aktif" if umur_bulan<=23 else "Lewat"
        st.markdown(f"""
        <div class="metarow">
            <div class="mpill">
                <div class="mplabel">BMI Anak</div>
                <div class="mpval">{bmi_val:.1f}</div>
                <div class="mpsub">kg / m²</div>
            </div>
            <div class="mpill">
                <div class="mplabel">Kelompok Usia</div>
                <div class="mpval" style="font-size:.95rem;padding:.3rem 0;">{kel_map.get(kel,'—')}</div>
                <div class="mpsub">Permenkes No.2/2020</div>
            </div>
            <div class="mpill">
                <div class="mplabel">Window 1000 HPK</div>
                <div class="mpval" style="color:{hpk_col};font-size:1.05rem;">{hpk_text}</div>
                <div class="mpsub">0 – 23 bulan kritis</div>
            </div>
            <div class="mpill">
                <div class="mplabel">Kepercayaan AI</div>
                <div class="mpval" style="color:{conf_col};">{confidence:.0f}%</div>
                <div class="mpsub">{conf_lbl}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── REKOMENDASI INTERVENSI
        border_col = rp["color"]
        recs_html = "".join([f'<div class="rec-item">{r}</div>' for r in rp["recs"]])
        st.markdown(f"""
        <div class="recbox" style="border-left-color:{border_col};">
            <span class="acc" style="color:{border_col};">Rekomendasi Intervensi — </span>
            <div style="margin-top:.6rem;">{recs_html}</div>
            <div class="disc">
                Hasil ini merupakan skrining awal berbasis AI dan tidak menggantikan
                diagnosis medis profesional. Konsultasikan ke tenaga kesehatan untuk
                penilaian klinis yang menyeluruh.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — GRAFIK PERTUMBUHAN
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📊"):
    st.markdown('<p class="slabel">WHO Growth Chart — Kurva Pertumbuhan Standar</p>', unsafe_allow_html=True)

    res = st.session_state.last_result
    if res:
        jk_lbl  = res["jk"]
        jk_code = "L" if jk_lbl=="Laki-laki" else "P"
        st.info(f"📌 Menggunakan data terakhir: **{res['nama'] or 'Anak'}** — "
                f"Usia {res['umur']} bln · BB {res['berat']} kg · TB {res['tinggi']} cm")
        umur_g=res["umur"]; berat_g=res["berat"]; tinggi_g=res["tinggi"]; jk_g=jk_code
    else:
        st.info("Belum ada prediksi. Gunakan form di bawah atau lakukan prediksi dulu.")
        c1,c2,c3 = st.columns(3)
        with c1: umur_g  = st.number_input("Umur (bln)", 0,60,24,1,key="g_umur")
        with c2: berat_g = st.number_input("Berat (kg)",1.0,35.0,12.0,.1,format="%.1f",key="g_berat")
        with c3: tinggi_g= st.number_input("Tinggi (cm)",40.0,130.0,87.0,.1,format="%.1f",key="g_tinggi")
        jk_g = "L" if st.selectbox("Jenis Kelamin",["Laki-laki","Perempuan"],key="g_jk")=="Laki-laki" else "P"

    fig = plot_growth_chart(umur_g, tinggi_g, berat_g, jk_g)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Interpretasi posisi anak
    # Interpolate median & -2SD for child's age
    hfa = HFA_BOYS if jk_g=="L" else HFA_GIRLS
    ages_hfa = [r[0] for r in hfa]
    idx = bisect.bisect_left(ages_hfa, umur_g)
    idx = max(1, min(len(hfa)-1, idx))
    t   = (umur_g - ages_hfa[idx-1])/(ages_hfa[idx]-ages_hfa[idx-1]+1e-9)
    med_tb  = hfa[idx-1][3]+t*(hfa[idx][3]-hfa[idx-1][3])
    s2n_tb  = hfa[idx-1][2]+t*(hfa[idx][2]-hfa[idx-1][2])
    s3n_tb  = hfa[idx-1][1]+t*(hfa[idx][1]-hfa[idx-1][1])
    gap_med = tinggi_g - med_tb
    gap_s2  = tinggi_g - s2n_tb

    st.markdown(f"""
    <div class="ibox" style="margin-top:1rem;">
        <div class="ibox-title">📊 Interpretasi Posisi Anak pada Kurva WHO</div>
        <div style="font-size:.88rem;color:#94a3b8;line-height:1.8;">
            <b style="color:#e8d48b;">Tinggi badan:</b> {tinggi_g:.1f} cm
            &nbsp;·&nbsp;
            <b style="color:#94a3b8;">Median WHO usia {umur_g} bln:</b> {med_tb:.1f} cm
            &nbsp;·&nbsp;
            <b style="color:#94a3b8;">Selisih dari median:</b>
            <span style="color:{'#34d399' if gap_med>=0 else '#f87171'};">{gap_med:+.1f} cm</span>
        </div>
        <div style="font-size:.85rem;color:#94a3b8;margin-top:.5rem;">
            {"✅ Tinggi badan berada di atas atau pada batas median WHO — pertumbuhan baik." if tinggi_g >= med_tb
             else ("⚠️ Tinggi badan berada antara −2 SD dan median — perlu dipantau." if tinggi_g >= s2n_tb
             else "🔴 Tinggi badan berada di bawah −2 SD — indikasi stunting pada standar TB/U.")}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("🧬"):
    st.markdown('<p class="slabel">Faktor Penentu Prediksi — Feature Importance</p>', unsafe_allow_html=True)

    fig = plot_feature_importance()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="ibox" style="margin-top:1rem;">
        <div class="ibox-title">🔎 Penjelasan Feature Importance</div>
        <div style="font-size:.87rem;color:#94a3b8;line-height:1.8;">
            Feature importance menunjukkan <b style="color:#e8d48b;">seberapa besar kontribusi tiap variabel</b>
            dalam keputusan model ensemble (CatBoost + XGBoost).
            Nilai dihitung dari rata-rata gain (CatBoost) dan weight (XGBoost) ternormalisasi.<br><br>
            <b style="color:#e8d48b;">Temuan utama:</b>
            Indeks gizi berbasis Z-Score (BB/TB & BB/U) mendominasi prediksi karena langsung
            mencerminkan status gizi anak relatif terhadap standar WHO. Tinggi badan raw
            berkontribusi signifikan karena menjadi pembagi dalam indeks BB/TB.
            Interaksi umur × z-score menangkap perbedaan risiko antar kelompok usia kritis (1000 HPK).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Table
    st.markdown('<p class="slabel" style="margin-top:1.2rem;">Tabel Lengkap</p>', unsafe_allow_html=True)
    df_fi = pd.DataFrame(FEATURE_IMPORTANCE, columns=["Fitur","Kontribusi (%)"])
    df_fi.index = df_fi.index + 1
    st.dataframe(
        df_fi.style.background_gradient(subset=["Kontribusi (%)"], cmap="YlOrRd"),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — SIMULASI RISIKO
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("⚡"):
    st.markdown('<p class="slabel">Simulasi — Bagaimana Jika Parameter Berubah?</p>', unsafe_allow_html=True)

    res = st.session_state.last_result
    if not res:
        st.warning("⚠️ Lakukan prediksi tunggal terlebih dahulu di menu Prediksi Tunggal.")
        st.stop()

    base_berat  = res["berat"]
    base_tinggi = res["tinggi"]
    base_umur   = res["umur"]
    jk          = "L" if res["jk"]=="Laki-laki" else "P"
    cara        = res["cara"]
    base_pct    = res["pct"]

    st.markdown(f"""
    <div class="gcard" style="margin-bottom:1.2rem;">
        <div class="ibox-title">📌 Data Dasar</div>
        <div style="font-size:.88rem;color:#94a3b8;">
            Usia: <b style="color:#e8d48b;">{base_umur} bln</b> &nbsp;·&nbsp;
            BB: <b style="color:#e8d48b;">{base_berat} kg</b> &nbsp;·&nbsp;
            TB: <b style="color:#e8d48b;">{base_tinggi} cm</b> &nbsp;·&nbsp;
            Prob dasar: <b style="color:{res['tier']['color']};">{base_pct:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        delta_bb  = st.slider("Δ Berat Badan (kg)", -3.0, 5.0, 0.0, 0.1,
                               format="%+.1f kg", help="Perubahan berat dari nilai dasar")
        delta_tb  = st.slider("Δ Tinggi / Panjang Badan (cm)", -5.0, 10.0, 0.0, 0.5,
                               format="%+.1f cm", help="Perubahan tinggi dari nilai dasar")
    with c2:
        delta_umur = st.slider("Δ Usia (bulan ke depan)", 0, 12, 0, 1,
                                format="+%d bln", help="Proyeksi pertumbuhan di masa mendatang")

    # Simulate with multiple delta points for chart
    with st.spinner("Menghitung simulasi..."):
        try:
            model, threshold, _ = load_model()
            new_berat  = max(1.0, base_berat  + delta_bb)
            new_tinggi = max(40.0, base_tinggi + delta_tb)
            new_umur   = min(60, base_umur + delta_umur)

            X_new, zs_new_bbu, zs_new_bbtb = build_features(new_umur, jk, new_berat, new_tinggi, cara)
            proba_new  = model.predict_proba(X_new)[0]
            pct_new    = proba_new[1] * 100
            tier_new   = risk_tier(pct_new)
            delta_pct  = pct_new - base_pct

            # Sensitivity curve: berat ±3
            bb_range = np.arange(max(1.0, base_berat-3), min(30, base_berat+5.1), 0.5)
            pct_bb = []
            for bb in bb_range:
                X_t,_,_ = build_features(new_umur, jk, bb, new_tinggi, cara)
                pct_bb.append(model.predict_proba(X_t)[0][1]*100)

        except Exception as e:
            st.error(f"Error simulasi: {e}"); st.stop()

    # Result
    arrow = "↓" if delta_pct < 0 else "↑"
    arr_col = "#34d399" if delta_pct < 0 else "#f87171"
    st.markdown(f"""
    <div class="rp {tier_new['cls']}" style="margin-top:1rem;">
        <div class="rp-eyebrow" style="color:{tier_new['color']};">Hasil Simulasi</div>
        <div class="rp-icon">{tier_new['icon']}</div>
        <div class="rp-headline" style="color:{tier_new['color']};">{tier_new['title']}</div>
        <div class="rp-prob">
            Probabilitas: <strong style="color:{tier_new['color']};">{pct_new:.1f}%</strong>
            &nbsp;
            <span style="color:{arr_col}; font-weight:600;">{arrow} {abs(delta_pct):.1f}%</span>
            dari data dasar ({base_pct:.1f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Z-score comparison
    c_a, c_b = st.columns(2)
    bc_new,bcol_new,bbg_new = bbu_style(zs_new_bbu)
    btbc_new,btbcol_new,btbbg_new = bbtb_style(zs_new_bbtb)
    with c_a:
        st.markdown(f"""
        <div class="zscard" style="text-align:center;padding:1rem;">
            <div class="zstag">Z-Score BB/U (Simulasi)</div>
            <div class="zsval" style="color:{bcol_new};">{zs_new_bbu:+.2f}</div>
            <span class="zsbadge" style="background:{bbg_new};color:{bcol_new};">{bc_new}</span>
        </div>""", unsafe_allow_html=True)
    with c_b:
        st.markdown(f"""
        <div class="zscard" style="text-align:center;padding:1rem;">
            <div class="zstag">Z-Score BB/TB (Simulasi)</div>
            <div class="zsval" style="color:{btbcol_new};">{zs_new_bbtb:+.2f}</div>
            <span class="zsbadge" style="background:{btbbg_new};color:{btbcol_new};">{btbc_new}</span>
        </div>""", unsafe_allow_html=True)

    # Sensitivity chart
    st.markdown('<p class="slabel" style="margin-top:1.2rem;">Kurva Sensitivitas — Probabilitas vs Berat Badan</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10,3.8))
    fig2.patch.set_facecolor("#1a2535"); ax2.set_facecolor("#111827")
    ax2.fill_between(bb_range, 45, 100, color="#f87171", alpha=.07)
    ax2.axhline(45, color="#facc15", lw=1, linestyle="--", alpha=.6, label="Batas 45%")
    ax2.plot(bb_range, pct_bb, color="#4fd1c5", lw=2.5)
    ax2.scatter([new_berat], [pct_new], color="#c9a84c", s=80, zorder=5, label=f"Simulasi ({new_berat:.1f} kg)")
    ax2.scatter([base_berat], [base_pct], color="#94a3b8", s=60, zorder=4, label=f"Data Dasar ({base_berat:.1f} kg)")
    ax2.set_xlabel("Berat Badan (kg)", color="#64748b", fontsize=9)
    ax2.set_ylabel("Probabilitas Stunting (%)", color="#64748b", fontsize=9)
    ax2.set_title("Sensitivitas Probabilitas Stunting terhadap Perubahan Berat Badan",
                  color="#e8d48b", fontsize=10, fontweight='normal')
    ax2.tick_params(colors="#64748b", labelsize=8.5)
    for sp in ['bottom','left']: ax2.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax2.spines[sp].set_visible(False)
    ax2.grid(color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(fontsize=8, facecolor="#1a2535", edgecolor="#1f2d42", labelcolor="#94a3b8")
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — RIWAYAT PREDIKSI
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📋"):
    st.markdown('<p class="slabel">Riwayat Prediksi — Sesi Ini</p>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi di menu Prediksi Tunggal.")
    else:
        df_hist = pd.DataFrame(hist)

        # Color coding
        def result_color(r):
            if "Normal" in r:   return "background-color:#0d2218;color:#34d399;"
            if "Diwaspadai" in r: return "background-color:#1a1a00;color:#facc15;"
            if "Sedang" in r:   return "background-color:#1a0e00;color:#fb923c;"
            return "background-color:#1a0909;color:#f87171;"

        # Summary stats
        total = len(df_hist)
        n_normal   = sum("Normal" in r for r in df_hist["Hasil"])
        n_risk     = total - n_normal
        avg_prob   = df_hist["Prob (%)"].mean()

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val,clr in [
            (c1,"Total Prediksi",str(total),"#e8d48b"),
            (c2,"Status Normal",str(n_normal),"#34d399"),
            (c3,"Indikasi Risiko",str(n_risk),"#f87171"),
            (c4,"Rata-rata Prob",f"{avg_prob:.1f}%","#4fd1c5"),
        ]:
            col.markdown(f"""
            <div class="stat-box">
                <div class="stat-val" style="color:{clr};">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # Table
        rows_html = ""
        for _, row in df_hist.iterrows():
            rst = row["Hasil"]
            if "Normal" in rst:       badge_style = "color:#34d399;background:rgba(52,211,153,.12);padding:.2rem .6rem;border-radius:6px;"
            elif "Diwaspadai" in rst: badge_style = "color:#facc15;background:rgba(250,204,21,.12);padding:.2rem .6rem;border-radius:6px;"
            elif "Sedang" in rst:     badge_style = "color:#fb923c;background:rgba(251,146,60,.12);padding:.2rem .6rem;border-radius:6px;"
            else:                     badge_style = "color:#f87171;background:rgba(248,113,113,.12);padding:.2rem .6rem;border-radius:6px;"
            rows_html += f"""
            <tr>
                <td>{row['Waktu']}</td>
                <td>{row['Nama']}</td>
                <td style="text-align:center;">{row['Umur (bln)']}</td>
                <td style="text-align:center;">{row['BB (kg)']}</td>
                <td style="text-align:center;">{row['TB (cm)']}</td>
                <td style="text-align:center;">{row['JK']}</td>
                <td style="text-align:center;color:#e8d48b;font-weight:500;">{row['Prob (%)']:.1f}%</td>
                <td><span style="{badge_style}">{rst}</span></td>
            </tr>"""

        st.markdown(f"""
        <div class="hist-wrap">
        <table>
            <thead><tr>
                <th>Waktu</th><th>Nama</th><th>Umur</th><th>BB</th>
                <th>TB</th><th>JK</th><th>Prob</th><th>Hasil</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

        # Export CSV
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Unduh Riwayat sebagai CSV",
            data=csv_data,
            file_name=f"riwayat_prediksi_{datetime.date.today()}.csv",
            mime="text/csv"
        )

        if st.button("🗑️  Hapus Riwayat"):
            st.session_state.history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — BATCH EXCEL UPLOAD
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📁"):
    st.markdown('<p class="slabel">Prediksi Massal — Upload File Excel</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ibox" style="margin-bottom:1.2rem;">
        <div class="ibox-title">📋 Format File Excel</div>
        <div style="font-size:.86rem;color:#94a3b8;line-height:1.8;">
            File harus memiliki kolom berikut (nama persis, huruf kecil):<br>
            <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                umur_bulan</code> &nbsp;
            <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                jk</code> (L/P) &nbsp;
            <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                berat</code> &nbsp;
            <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                tinggi</code> &nbsp;
            <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                cara_ukur</code> (Berdiri/Terlentang)
            <br>Kolom <code style="background:#0b1120;color:#4fd1c5;padding:.15rem .4rem;border-radius:4px;">
                nama</code> bersifat opsional.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Template download
    template_df = pd.DataFrame({
        "nama":       ["Budi","Sari"],
        "umur_bulan": [24, 36],
        "jk":         ["L","P"],
        "berat":      [11.5, 13.2],
        "tinggi":     [85.0, 93.5],
        "cara_ukur":  ["Berdiri","Berdiri"]
    })
    tpl_bytes = io.BytesIO()
    template_df.to_excel(tpl_bytes, index=False)
    st.download_button(
        "⬇️  Unduh Template Excel",
        data=tpl_bytes.getvalue(),
        file_name="template_prediksi_stunting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx","xls"])
    if uploaded:
        try:
            df_up = pd.read_excel(uploaded)
            df_up.columns = df_up.columns.str.lower().str.strip()
            required = ["umur_bulan","jk","berat","tinggi","cara_ukur"]
            missing  = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {missing}")
            else:
                model, threshold, _ = load_model()
                results = []
                prog = st.progress(0)
                for idx, row in df_up.iterrows():
                    prog.progress((idx+1)/len(df_up))
                    try:
                        X, z_bbu, z_bbtb = build_features(
                            int(row.umur_bulan), str(row.jk).strip(),
                            float(row.berat), float(row.tinggi),
                            str(row.cara_ukur).strip()
                        )
                        prob = model.predict_proba(X)[0][1]*100
                        tier = risk_tier(prob)
                        bc,bcol,_ = bbu_style(z_bbu); btbc,_,_ = bbtb_style(z_bbtb)
                        results.append({
                            "Nama":     row.get("nama","—"),
                            "Umur":     int(row.umur_bulan),
                            "JK":       row.jk,
                            "BB (kg)":  float(row.berat),
                            "TB (cm)":  float(row.tinggi),
                            "ZS BB/U":  round(z_bbu,2),
                            "ZS BB/TB": round(z_bbtb,2),
                            "Status BB/U": bc,
                            "Status BB/TB": btbc,
                            "Prob (%)": round(prob,1),
                            "Hasil":    tier["title"],
                            "Level":    tier["level"],
                        })
                    except Exception as ex:
                        results.append({"Nama": row.get("nama","?"), "Error": str(ex)})
                prog.empty()

                df_out = pd.DataFrame(results)
                # Summary
                total_b = len(df_out)
                n_ok    = sum("Normal" in str(r) for r in df_out.get("Hasil",[]))
                n_risk2 = total_b - n_ok
                st.markdown(f"""
                <div class="metarow" style="margin-top:1rem;">
                    <div class="mpill"><div class="mplabel">Total Data</div>
                        <div class="mpval">{total_b}</div><div class="mpsub">anak dianalisis</div></div>
                    <div class="mpill"><div class="mplabel">Normal</div>
                        <div class="mpval" style="color:#34d399;">{n_ok}</div>
                        <div class="mpsub">aman</div></div>
                    <div class="mpill"><div class="mplabel">Perlu Perhatian</div>
                        <div class="mpval" style="color:#f87171;">{n_risk2}</div>
                        <div class="mpsub">ada indikasi</div></div>
                    <div class="mpill"><div class="mplabel">Prevalensi</div>
                        <div class="mpval" style="color:#facc15;">{n_risk2/total_b*100:.1f}%</div>
                        <div class="mpsub">dari total</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(df_out, use_container_width=True)
                csv_out = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Unduh Hasil sebagai CSV",
                    data=csv_out,
                    file_name=f"hasil_batch_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error memproses file: {e}")


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — TENTANG MODEL
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📚"):
    st.markdown('<p class="slabel">Tentang Model & Metodologi</p>', unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-val">92.7%</div>
            <div class="stat-lbl">Akurasi</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">91.4%</div>
            <div class="stat-lbl">Precision</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">90.3%</div>
            <div class="stat-lbl">Recall</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">96.3%</div>
            <div class="stat-lbl">AUC-ROC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric bar chart
    fig3, ax3 = plt.subplots(figsize=(9, 3))
    fig3.patch.set_facecolor("#1a2535"); ax3.set_facecolor("#111827")
    metrics_lbl = list(MODEL_METRICS.keys())
    metrics_val = [v*100 for v in MODEL_METRICS.values()]
    clrs = ["#c9a84c","#4fd1c5","#34d399","#fb923c","#a78bfa"]
    bars3 = ax3.bar(metrics_lbl, metrics_val, color=clrs, edgecolor="none", width=0.5)
    for bar, val in zip(bars3, metrics_val):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{val:.1f}%", ha='center', va='bottom', color="#94a3b8", fontsize=9)
    ax3.set_ylim(80, 100)
    ax3.set_title("Performa Model pada Test Set (20% Data)", color="#e8d48b", fontsize=10, fontweight='normal')
    ax3.tick_params(colors="#64748b", labelsize=9)
    for sp in ['bottom','left']: ax3.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax3.spines[sp].set_visible(False)
    ax3.grid(axis='y', color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Info boxes
    t1, t2 = st.tabs(["🏗️  Arsitektur Model", "📋  Metodologi & Dasar Hukum"])

    with t1:
        st.markdown("""
        <div class="ibox">
        <div class="ibox-title">Ensemble: CatBoost + XGBoost</div>
        <div style="font-size:.87rem;color:#94a3b8;line-height:1.9;">
        <b style="color:#e8d48b;">CatBoost (bobot 45%)</b><br>
        &nbsp;· iterations=3000, learning_rate=0.01, depth=7, l2_leaf_reg=5<br>
        &nbsp;· class_weights=[1, 2.0], rsm=0.8, bagging_temperature=0.5<br>
        &nbsp;· Dioptimasi dengan early stopping (patience=200)<br><br>
        <b style="color:#e8d48b;">XGBoost (bobot 55%)</b><br>
        &nbsp;· n_estimators=2000, learning_rate=0.02, max_depth=7<br>
        &nbsp;· reg_lambda=5, scale_pos_weight=4<br>
        &nbsp;· Early stopping rounds=100<br><br>
        <b style="color:#e8d48b;">Penanganan Imbalanced Data</b><br>
        &nbsp;· Teknik SMOTETomek untuk oversampling kelas minoritas stunting<br>
        &nbsp;· Threshold optimal dicari melalui optimasi F1-Score (bukan default 0.5)<br><br>
        <b style="color:#e8d48b;">Validasi</b><br>
        &nbsp;· 5-Fold Stratified Cross-Validation pada data asli (tanpa SMOTE)<br>
        &nbsp;· Train/Test split 80/20 dengan stratifikasi
        </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="ibox">
        <div class="ibox-title">Dasar Hukum & Referensi</div>
        <div style="font-size:.87rem;color:#94a3b8;line-height:1.9;">
        <b style="color:#e8d48b;">Permenkes No. 2 Tahun 2020</b> — Standar Antropometri Anak<br>
        Indeks yang digunakan:<br>
        &nbsp;· BB/U (Berat Badan menurut Umur) → menilai status gizi umum<br>
        &nbsp;· TB/U (Tinggi Badan menurut Umur) → TARGET (stunting = ZS &lt; −2 SD)<br>
        &nbsp;· BB/TB (Berat Badan menurut Tinggi) → menilai wasting<br><br>
        <b style="color:#e8d48b;">WHO Multicentre Growth Reference Study Group (2006)</b><br>
        &nbsp;· Tabel referensi LMS untuk Z-Score BB/U dan BB/TB<br>
        &nbsp;· Kurva pertumbuhan Height-for-Age standar internasional<br><br>
        <b style="color:#e8d48b;">Feature Engineering (70+ Variabel)</b><br>
        &nbsp;· Kelompok usia Permenkes: ASI Eksklusif, MPASI, Baduta, Batita, Balita<br>
        &nbsp;· Window 1000 HPK (0–23 bulan) sebagai periode kritis<br>
        &nbsp;· Interaksi umur × gizi, jenis kelamin × z-score<br>
        &nbsp;· Indeks komposit risiko gizi, zona borderline, proximity ke SD ambang
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ibox" style="margin-top:.8rem;">
        <div class="ibox-title">⚠️ Disclaimer</div>
        <div style="font-size:.85rem;color:#94a3b8;line-height:1.8;">
        Aplikasi ini merupakan <b style="color:#e8d48b;">alat skrining awal berbasis kecerdasan buatan</b>
        dan <b style="color:#f87171;">tidak menggantikan diagnosis medis profesional</b>.
        Keputusan klinis harus selalu dikonsultasikan dengan dokter, bidan, atau tenaga
        kesehatan yang kompeten. Hasil prediksi dipengaruhi oleh akurasi data yang dimasukkan.
        </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fnote">
    <span>Permenkes No. 2 Tahun 2020</span> — Standar Antropometri Anak
    &nbsp;·&nbsp; <span>WHO 2006 Multicentre Growth Reference Study</span>
    <br>Model Ensemble CatBoost + XGBoost &nbsp;·&nbsp; Skrining Awal — Bukan Pengganti Diagnosis Medis
</div>
""", unsafe_allow_html=True)

