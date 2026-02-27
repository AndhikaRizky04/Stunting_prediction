import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bisect
import io
import datetime
import scipy.stats as stats_scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkriningGizi · Prediksi Stunting",
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
    --teal:#4fd1c5; --purple:#a78bfa;
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

/* ── TOP BAR ── */
.topbar{
    display:flex; align-items:center; gap:1rem;
    padding:1rem 1.6rem; margin-bottom:1.8rem;
    background:var(--navy-3); border:1px solid var(--border-gold);
    border-radius:16px;
}
.topbar-icon{font-size:1.6rem;line-height:1;}
.topbar-title{font-family:'Cormorant Garamond',serif;font-size:1.55rem;font-weight:300;color:var(--text-1);}
.topbar-title em{font-style:italic;color:var(--gold-lt);}
.topbar-sub{font-size:.73rem;color:var(--text-3);letter-spacing:.04em;margin-top:.1rem;}
.topbar-right{margin-left:auto;text-align:right;}
.topbar-badge{
    display:inline-block; background:var(--gold-dim);
    border:1px solid var(--border-gold); color:var(--gold-lt);
    font-size:.62rem; font-weight:600; letter-spacing:.12em;
    text-transform:uppercase; padding:.28rem .75rem; border-radius:100px;
}

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
div[data-testid="stSelectbox"]>div>div,
div[data-testid="stTextInput"] input{
    background:var(--navy-3)!important; border:1.5px solid rgba(255,255,255,.1)!important;
    border-radius:12px!important; color:var(--text-1)!important;
    font-family:'Outfit',sans-serif!important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"]>div>div:focus-within,
div[data-testid="stTextInput"] input:focus{
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

/* ── CTA BUTTON ── */
.stButton>button{
    width:100%!important;
    background:linear-gradient(135deg,#9a6e1a,var(--gold) 50%,var(--gold-lt))!important;
    color:#0b1120!important; border:none!important; border-radius:14px!important;
    padding:.9rem 2rem!important; font-size:.9rem!important; font-weight:600!important;
    letter-spacing:.07em!important; text-transform:uppercase!important;
    font-family:'Outfit',sans-serif!important;
    box-shadow:0 8px 24px rgba(201,168,76,.28)!important;
    transition:all .3s cubic-bezier(.4,0,.2,1)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 12px 32px rgba(201,168,76,.38)!important;}

/* ── RESULT PANELS ── */
.rp{border-radius:20px;padding:1.8rem;text-align:center;position:relative;overflow:hidden;animation:resultPop .5s cubic-bezier(.34,1.4,.64,1) both;}
.rp::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 65% 45% at 50% 0%,var(--rp-glow) 0%,transparent 70%);}
.rp-normal {background:linear-gradient(160deg,#091a12,#0d2218);border:1px solid rgba(52,211,153,.25);--rp-glow:rgba(52,211,153,.12);}
.rp-waspada{background:linear-gradient(160deg,#1a1400,#221a00);border:1px solid rgba(250,204,21,.25);--rp-glow:rgba(250,204,21,.10);}
.rp-sedang {background:linear-gradient(160deg,#1a0e00,#221400);border:1px solid rgba(251,146,60,.25);--rp-glow:rgba(251,146,60,.11);}
.rp-tinggi {background:linear-gradient(160deg,#1a0909,#220d0d);border:1px solid rgba(248,113,113,.25);--rp-glow:rgba(248,113,113,.12);}
.rp-eyebrow{font-size:.63rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;opacity:.7;margin-bottom:.6rem;position:relative;}
.rp-icon{font-size:2.2rem;margin-bottom:.3rem;position:relative;line-height:1;}
.rp-headline{font-family:'Cormorant Garamond',serif;font-size:2.2rem;font-weight:600;letter-spacing:-.03em;line-height:1;margin-bottom:.35rem;position:relative;}
.rp-prob{font-size:.82rem;font-weight:400;opacity:.65;position:relative;}

/* ── Z-SCORE GRID (3-column) ── */
.zsgrid3{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-bottom:1rem;}
.zsgrid2{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin-bottom:1rem;}
.zscard{background:var(--navy-3);border:1px solid var(--border);border-radius:14px;padding:1.1rem;text-align:center;transition:border-color .2s,transform .15s;}
.zscard:hover{border-color:var(--border-gold);transform:translateY(-2px);}
.zstag{font-size:.6rem;letter-spacing:.13em;text-transform:uppercase;font-weight:600;color:var(--text-3);margin-bottom:.5rem;}
.zsval{font-family:'Cormorant Garamond',serif;font-size:2.4rem;font-weight:300;line-height:1;margin-bottom:.3rem;}
.zsdesc{font-size:.68rem;color:var(--text-3);margin-bottom:.4rem;}
.zsbadge{font-size:.7rem;font-weight:500;padding:.22rem .65rem;border-radius:100px;display:inline-block;}
.zspct{font-size:.68rem;color:var(--text-3);margin-top:.3rem;}

/* ── GAUGE ── */
.gwrap{background:var(--navy-3);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem;}
.gtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.8rem;}
.gttl{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;font-weight:600;color:var(--text-3);}
.gnum{font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:300;line-height:1;}
.gtrack{height:6px;border-radius:100px;background:rgba(255,255,255,.07);position:relative;}
.gbar{position:absolute;left:0;top:0;height:100%;border-radius:100px;background:linear-gradient(90deg,#34d399 0%,#facc15 45%,#fb923c 70%,#f87171 100%);}
.gpip{position:absolute;top:50%;transform:translate(-50%,-50%);width:12px;height:12px;border-radius:50%;border:2px solid var(--navy);z-index:2;box-shadow:0 0 8px currentColor;}
.gtick{display:flex;justify-content:space-between;margin-top:.4rem;font-size:.65rem;color:var(--text-3);}

/* ── META PILLS ── */
.metarow{display:flex;gap:.7rem;margin-bottom:1rem;flex-wrap:wrap;}
.mpill{flex:1;min-width:100px;background:var(--navy-3);border:1px solid var(--border);border-radius:12px;padding:.8rem .6rem;text-align:center;transition:border-color .2s;}
.mpill:hover{border-color:var(--border-gold);}
.mplabel{font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--text-3);font-weight:600;}
.mpval{font-family:'Cormorant Garamond',serif;font-size:1.3rem;font-weight:300;color:var(--text-1);margin:.18rem 0 .1rem;line-height:1;}
.mpsub{font-size:.64rem;color:var(--text-3);}

/* ── THREE-INDEX STATUS TABLE ── */
.tistable{width:100%;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;border:1px solid var(--border);margin-bottom:1rem;}
.tistable thead tr{background:var(--navy-4);}
.tistable th{padding:.6rem .9rem;font-size:.68rem;color:var(--text-3);font-weight:600;letter-spacing:.1em;text-transform:uppercase;text-align:left;}
.tistable td{padding:.7rem .9rem;font-size:.84rem;color:var(--text-2);border-top:1px solid var(--border);}
.tistable .key-index{color:var(--gold-lt);font-weight:500;}
.tistable .stunting-row td{background:rgba(248,113,113,.06);}

/* ── INTERPRETATION BOX ── */
.ibox{background:var(--navy-3);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:14px;padding:1rem 1.2rem;margin-bottom:.9rem;}
.ibox-title{font-size:.64rem;letter-spacing:.15em;text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:.7rem;}
.irow{display:flex;align-items:center;gap:.7rem;padding:.4rem .5rem;border-radius:8px;margin-bottom:.25rem;font-size:.85rem;}
.irow.active{border:1px solid;}
.irow-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}

/* ── RECOMMENDATION BOX ── */
.recbox{background:var(--navy-3);border:1px solid var(--border);border-left:3px solid;border-radius:14px;padding:1rem 1.2rem;font-size:.86rem;line-height:1.7;color:var(--text-2);margin-bottom:.9rem;}
.recbox .acc{font-weight:500;}
.recbox .disc{font-size:.74rem;color:var(--text-3);margin-top:.5rem;font-style:italic;}
.rec-item{display:flex;align-items:flex-start;gap:.5rem;margin:.22rem 0;}
.rec-item::before{content:'→';color:var(--gold);flex-shrink:0;font-size:.88rem;}

/* ── CONFIDENCE BADGE ── */
.conf-badge{display:inline-flex;align-items:center;gap:.5rem;padding:.35rem .9rem;border-radius:100px;border:1px solid;font-size:.78rem;font-weight:500;margin-bottom:.8rem;}

/* ── TRACKING CARD ── */
.track-card{background:var(--navy-3);border:1px solid var(--border);border-radius:14px;padding:1.1rem;margin-bottom:.7rem;display:flex;align-items:center;gap:1rem;transition:border-color .2s;}
.track-card:hover{border-color:var(--border-gold);}
.track-date{font-size:.7rem;color:var(--text-3);min-width:70px;}
.track-vals{flex:1;display:flex;gap:1.2rem;flex-wrap:wrap;}
.track-val{text-align:center;}
.track-val-num{font-family:'Cormorant Garamond',serif;font-size:1.2rem;color:var(--text-1);line-height:1;}
.track-val-lbl{font-size:.62rem;color:var(--text-3);letter-spacing:.08em;text-transform:uppercase;}
.track-prob{min-width:60px;text-align:right;}

/* ── HISTORY TABLE ── */
.hist-wrap{overflow-x:auto;border-radius:14px;border:1px solid var(--border);}
.hist-wrap table{width:100%;border-collapse:collapse;font-size:.83rem;}
.hist-wrap th{background:var(--navy-4);color:var(--text-3);font-weight:500;padding:.65rem .9rem;text-align:left;letter-spacing:.05em;font-size:.7rem;text-transform:uppercase;}
.hist-wrap td{padding:.6rem .9rem;border-bottom:1px solid var(--border);color:var(--text-2);}
.hist-wrap tr:last-child td{border-bottom:none;}
.hist-wrap tr:hover td{background:rgba(255,255,255,.02);}

/* ── ALERTS ── */
div[data-testid="stAlert"]{background:rgba(201,168,76,.08)!important;border:1px solid var(--border-gold)!important;border-radius:12px!important;color:var(--gold-lt)!important;font-family:'Outfit',sans-serif!important;}
div[data-testid="stSpinner"] p{color:var(--text-2)!important;font-family:'Outfit',sans-serif!important;}

/* ── TABS ── */
div[data-testid="stTabs"] [role="tablist"]{gap:.3rem;border-bottom:1px solid var(--border-gold)!important;}
div[data-testid="stTabs"] [role="tab"]{
    background:transparent!important; border:none!important;
    color:var(--text-3)!important; font-family:'Outfit',sans-serif!important;
    font-size:.82rem!important; font-weight:500!important;
    padding:.5rem .9rem!important; border-radius:8px 8px 0 0!important;
    letter-spacing:.03em!important;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
    background:var(--gold-dim)!important; color:var(--gold-lt)!important;
    border-bottom:2px solid var(--gold)!important;
}
div[data-testid="stTabs"] [role="tab"]:hover{color:var(--text-1)!important;background:rgba(255,255,255,.04)!important;}

/* ── PREMATURE TAG ── */
.prem-tag{display:inline-flex;align-items:center;gap:.35rem;background:rgba(167,139,250,.12);border:1px solid rgba(167,139,250,.3);color:#a78bfa;font-size:.73rem;font-weight:500;padding:.28rem .75rem;border-radius:100px;}

/* ── FOOTER ── */
.fnote{text-align:center;color:var(--text-3);font-size:.72rem;line-height:1.9;padding-top:1.2rem;margin-top:2rem;border-top:1px solid var(--border);}
.fnote span{color:var(--text-2);}

/* ── ANIMATIONS ── */
@keyframes fadeDown{from{opacity:0;transform:translateY(-14px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
@keyframes resultPop{from{opacity:0;transform:scale(.94);}to{opacity:1;transform:scale(1);}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WHO LMS TABLES — BB/U
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

# ─────────────────────────────────────────────
# WHO LMS TABLES — BB/TB (Wasting)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# WHO SD TABLES — TB/U (Stunting indicator)
# [age_months, -3SD, -2SD, median, +2SD, +3SD]
# ─────────────────────────────────────────────
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
# FEATURE IMPORTANCE (from training)
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

# ─────────────────────────────────────────────
# Z-SCORE FUNCTIONS
# ─────────────────────────────────────────────
def lms_zscore(X, L, M, S):
    if L == 0:
        z = np.log(X / M) / S
    else:
        z = ((X / M) ** L - 1) / (L * S)
    if z > 3:
        SD3pos  = M * (1 + L*S*3)**(1/L)
        SD23pos = SD3pos - M*(1+L*S*2)**(1/L)
        z = 3 + (X - SD3pos)/SD23pos if SD23pos != 0 else 3
    elif z < -3:
        SD3neg  = M * (1+L*S*(-3))**(1/L)
        SD23neg = M*(1+L*S*(-2))**(1/L) - SD3neg
        z = -3 + (X - SD3neg)/SD23neg if SD23neg != 0 else -3
    return round(z, 2)

def _interp_lms_age(age, table):
    ages = [r[0] for r in table]
    a = max(0, min(60, int(round(age))))
    idx = bisect.bisect_left(ages, a)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0 and ages[idx] != a:
        a0, a1 = ages[idx-1], ages[idx]
        t = (a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,4))
    return table[idx][1], table[idx][2], table[idx][3]

def _interp_lms_height(h, table):
    hs = [r[0] for r in table]
    h  = max(hs[0], min(hs[-1], h))
    idx = bisect.bisect_left(hs, h)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0:
        h0, h1 = hs[idx-1], hs[idx]
        t = (h-h0)/(h1-h0) if h1!=h0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,4))
    return table[idx][1], table[idx][2], table[idx][3]

def _interp_hfa(age, table):
    """Returns (s3n, s2n, median, s2p, s3p) for given age by interpolation."""
    ages = [r[0] for r in table]
    a    = max(0, min(60, age))
    idx  = bisect.bisect_left(ages, a)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0 and ages[idx] != a:
        a0, a1 = ages[idx-1], ages[idx]
        t = (a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,6))
    return tuple(table[idx][1:6])

def zscore_bbu(weight, age, sex):
    L, M, S = _interp_lms_age(age, WFA_BOYS if sex=='L' else WFA_GIRLS)
    return lms_zscore(weight, L, M, S)

def zscore_bbtb(weight, h, sex, cara, age):
    hh = h
    if cara=='Terlentang' and age>=24: hh = h - 0.7
    elif cara=='Berdiri' and age<24:   hh = h + 0.7
    L, M, S = _interp_lms_height(hh, WFL_BOYS if sex=='L' else WFL_GIRLS)
    return lms_zscore(weight, L, M, S)

def zscore_tbu(height, age, sex):
    """TB/U z-score (Stunting indicator) using WHO SD table method."""
    s3n, s2n, med, s2p, s3p = _interp_hfa(age, HFA_BOYS if sex=='L' else HFA_GIRLS)
    if height >= med:
        sd_pos = (s2p - med) / 2
        z = (height - med) / sd_pos if sd_pos > 0 else 0
        if z > 3:
            sd23 = s3p - s2p
            z = 3 + (height - s3p) / sd23 if sd23 > 0 else 3
    else:
        sd_neg = (med - s2n) / 2
        z = (height - med) / sd_neg if sd_neg > 0 else 0
        if z < -3:
            sd23 = s2n - s3n
            z = -3 - (s3n - height) / sd23 if sd23 > 0 else -3
    return round(z, 2)

def zscore_to_percentile(z):
    """Convert Z-score to approximate WHO percentile."""
    return round(stats_scipy.norm.cdf(z) * 100, 1)

def median_tbu(age, sex):
    _, _, med, _, _ = _interp_hfa(age, HFA_BOYS if sex=='L' else HFA_GIRLS)
    return round(med, 1)

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
# STYLE HELPERS
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
                "Stimulasi tumbuh kembang dengan bermain aktif dan interaktif",
            ]
        )
    elif pct < 65:
        return dict(
            cls="rp-waspada", icon="○", title="Perlu Diwaspadai",
            color="#facc15", eyebrow="Status Pertumbuhan · Perlu Perhatian",
            level="Sedang-Rendah", badge_bg="rgba(250,204,21,.15)", badge_col="#facc15",
            recs=[
                "Konsultasikan ke petugas gizi atau bidan untuk evaluasi menyeluruh",
                "Pastikan kecukupan asupan protein hewani, zat besi, dan zinc",
                "Pantau berat dan tinggi badan setiap bulan secara konsisten",
                "Perhatikan diversifikasi pola makan dan jadwal makan anak",
            ]
        )
    elif pct < 80:
        return dict(
            cls="rp-sedang", icon="⚡", title="Risiko Stunting Sedang",
            color="#fb923c", eyebrow="Perhatian · Intervensi Gizi Dianjurkan",
            level="Sedang-Tinggi", badge_bg="rgba(251,146,60,.15)", badge_col="#fb923c",
            recs=[
                "Segera konsultasikan ke dokter atau ahli gizi anak terdekat",
                "Evaluasi pola makan, asupan kalori dan protein harian secara detail",
                "Pertimbangkan suplemen gizi mikro: vitamin A, zinc, dan zat besi",
                "Pemantauan pertumbuhan setiap 2 minggu hingga kondisi membaik",
            ]
        )
    else:
        return dict(
            cls="rp-tinggi", icon="⚠️", title="Risiko Stunting Tinggi",
            color="#f87171", eyebrow="Perhatian · Tindak Lanjut Segera",
            level="Tinggi", badge_bg="rgba(248,113,113,.15)", badge_col="#f87171",
            recs=[
                "Bawa ke puskesmas atau dokter spesialis anak segera hari ini",
                "Diperlukan intervensi gizi intensif, terstruktur, dan berkelanjutan",
                "Evaluasi faktor penyebab: infeksi berulang, MPASI tidak adekuat, BBLR",
                "Pemantauan pertumbuhan setiap minggu selama program intervensi",
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

def tbu_style(z):
    """TB/U — Stunting status per Permenkes No. 2 / 2020."""
    if z < -3: return "Sangat Pendek","#f87171","rgba(248,113,113,.14)","Severely Stunted"
    if z < -2: return "Pendek","#fb923c","rgba(251,146,60,.14)","Stunted"
    if z <= 3: return "Normal","#34d399","rgba(52,211,153,.14)","Normal"
    return "Tinggi","#4fd1c5","rgba(79,209,197,.14)","Tall"

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
    berat_min = {0:2.0,6:5.5,12:7.0,24:9.0,36:11.0,48:13.0,60:15.0}
    berat_max = {0:5.5,6:10.5,12:14.0,24:17.0,36:20.0,48:23.0,60:27.0}
    for k in sorted(berat_min.keys()):
        if umur >= k: bmin, bmax = berat_min[k], berat_max[k]
    if berat < bmin:
        warnings.append(f"⚠️ BB {berat} kg kemungkinan terlalu rendah untuk usia {umur} bln (min ~{bmin} kg)")
    if berat > bmax:
        warnings.append(f"⚠️ BB {berat} kg kemungkinan terlalu tinggi untuk usia {umur} bln (maks ~{bmax} kg)")
    tinggi_min = {0:44,6:60,12:68,24:78,36:86,48:93,60:100}
    tinggi_max = {0:57,6:74,12:84,24:97,36:107,48:116,60:123}
    for k in sorted(tinggi_min.keys()):
        if umur >= k: tmin, tmax = tinggi_min[k], tinggi_max[k]
    if tinggi < tmin:
        warnings.append(f"⚠️ TB {tinggi} cm tidak realistis untuk usia {umur} bln (min ~{tmin} cm)")
    if tinggi > tmax:
        warnings.append(f"⚠️ TB {tinggi} cm tidak realistis untuk usia {umur} bln (maks ~{tmax} cm)")
    bmi = berat / (tinggi/100)**2
    if bmi < 10: warnings.append("⚠️ BMI sangat rendah — periksa kembali data berat dan tinggi")
    if bmi > 30: warnings.append("⚠️ BMI sangat tinggi — periksa kembali data berat dan tinggi")
    return warnings

# ─────────────────────────────────────────────
# WHO GROWTH CHART
# ─────────────────────────────────────────────
def plot_growth_chart(umur, tinggi, berat, sex):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("#1a2535")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#64748b", labelsize=8.5)
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
    ax.axvline(x=umur,  color="#c9a84c", lw=0.8, linestyle=":", alpha=.5)
    ax.axhline(y=tinggi,color="#c9a84c", lw=0.8, linestyle=":", alpha=.5)
    ax.set_xlabel("Umur (Bulan)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Tinggi/Panjang Badan (cm)", color="#94a3b8", fontsize=9)
    ax.set_title(f"TB/U — Tinggi menurut Umur · {'L' if sex=='L' else 'P'}",
                 color="#e8d48b", fontsize=10, fontweight='normal', pad=10)
    p1 = mpatches.Patch(facecolor="#34d399", alpha=.3, label="Normal (−2 s.d. +2 SD)")
    p2 = mpatches.Patch(facecolor="#facc15", alpha=.3, label="Borderline (±2~±3 SD)")
    p3 = mpatches.Patch(facecolor="#f87171", alpha=.3, label="Kritis (< −3 SD)")
    ax.legend(handles=[p1,p2,p3,
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#c9a84c',markersize=7,label='Posisi Anak')],
        fontsize=7.5, facecolor="#1a2535", edgecolor="#1f2d42",
        labelcolor="#94a3b8", loc="upper left")

    # BB/U chart
    wfa = WFA_BOYS if sex=="L" else WFA_GIRLS
    ages_w = [r[0] for r in wfa]
    meds_w = [r[2] for r in wfa]
    s2n_w=[]; s2p_w=[]; s3n_w=[]; s3p_w=[]
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
    ax2.axvline(x=umur,  color="#c9a84c", lw=0.8, linestyle=":", alpha=.5)
    ax2.axhline(y=berat, color="#c9a84c", lw=0.8, linestyle=":", alpha=.5)
    ax2.set_xlabel("Umur (Bulan)", color="#94a3b8", fontsize=9)
    ax2.set_ylabel("Berat Badan (kg)", color="#94a3b8", fontsize=9)
    ax2.set_title(f"BB/U — Berat menurut Umur · {'L' if sex=='L' else 'P'}",
                  color="#e8d48b", fontsize=10, fontweight='normal', pad=10)
    ax2.legend(handles=[p1,p2,p3,
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#c9a84c',markersize=7,label='Posisi Anak')],
        fontsize=7.5, facecolor="#1a2535", edgecolor="#1f2d42",
        labelcolor="#94a3b8", loc="upper left")

    plt.tight_layout(pad=2.5)
    return fig

def plot_tracking_chart(records, sex):
    """Plot growth trajectory for a child with multiple measurements."""
    if len(records) < 2:
        return None
    dates  = [r["umur"] for r in records]
    berats = [r["berat"] for r in records]
    tinggis= [r["tinggi"] for r in records]
    probs  = [r["prob"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#1a2535")

    titles = ["Berat Badan (kg)", "Tinggi/Panjang (cm)", "Probabilitas Stunting (%)"]
    data_sets = [berats, tinggis, probs]
    colors = ["#4fd1c5", "#c9a84c", "#f87171"]

    for i, (ax, vals, title, col) in enumerate(zip(axes, data_sets, titles, colors)):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#64748b", labelsize=8.5)
        for sp in ['bottom','left']: ax.spines[sp].set_color("#1f2d42")
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        ax.grid(color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.plot(dates, vals, "-o", color=col, lw=2.2, markersize=6, zorder=3)
        ax.fill_between(dates, vals, alpha=0.12, color=col)
        # annotate last point
        ax.annotate(f"{vals[-1]:.1f}", (dates[-1], vals[-1]),
                    textcoords="offset points", xytext=(5, 5),
                    color=col, fontsize=8.5, fontweight='600')
        if i == 2:  # prob chart
            ax.axhline(45, color="#facc15", lw=1, linestyle="--", alpha=.6, label="Batas 45%")
            ax.axhspan(0, 45, alpha=.04, color="#34d399")
            ax.axhspan(45, 100, alpha=.04, color="#f87171")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=7.5, facecolor="#1a2535", edgecolor="#1f2d42", labelcolor="#94a3b8")
        ax.set_xlabel("Umur (Bulan)", color="#94a3b8", fontsize=8.5)
        ax.set_title(title, color="#e8d48b", fontsize=9.5, fontweight='normal', pad=8)

    plt.tight_layout(pad=2)
    return fig

def plot_feature_importance():
    feats = FEATURE_IMPORTANCE[:12]
    names = [f[0] for f in feats]
    vals  = [f[1] for f in feats]
    colors = ["#c9a84c" if v>=12 else ("#4fd1c5" if v>=7 else ("#94a3b8" if v>=4 else "#475569")) for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#1a2535")
    ax.set_facecolor("#111827")
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.62, edgecolor="none")
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va='center', ha='left', color="#94a3b8", fontsize=8.5)
    ax.set_xlabel("Kontribusi (%)", color="#64748b", fontsize=9)
    ax.set_title("Feature Importance — Faktor Penentu Prediksi",
                 color="#e8d48b", fontsize=11, fontweight='normal', pad=12)
    ax.tick_params(colors="#64748b", labelsize=8.5)
    for sp in ['bottom','left']: ax.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    ax.set_xlim(0, max(vals)+4)
    ax.grid(axis='x', color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.8)
    plt.tight_layout(pad=2)
    return fig

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "tracking" not in st.session_state:
    st.session_state.tracking = {}   # {nama: [records]}

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1.5rem;'>
        <div style='font-size:1.8rem;margin-bottom:.3rem;'>🌿</div>
        <div style='font-family:"Cormorant Garamond",serif;font-size:1.15rem;color:#e8d48b;font-weight:300;'>
            SkriningGizi
        </div>
        <div style='font-size:.68rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-top:.15rem;'>
            Sistem Prediksi Stunting Balita
        </div>
    </div>
    <hr style='border:none;border-top:1px solid rgba(201,168,76,.2);margin-bottom:1.2rem;'>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Prediksi & Analisis",
         "📈  Rekam Tumbuh Kembang",
         "📊  Grafik Pertumbuhan WHO",
         "🧬  Analisis Fitur",
         "⚡  Simulasi Intervensi",
         "📋  Riwayat Prediksi",
         "📁  Prediksi Batch (Excel)"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:1.5rem 0 .8rem;'>
    <div style='font-size:.68rem;color:#334155;text-align:center;line-height:1.8;'>
        WHO 2006 Growth Standards<br>
        Permenkes No. 2 / 2020<br>
        Ensemble CatBoost + XGBoost
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP BAR (replaces elaborate hero)
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-icon">🌿</div>
    <div>
        <div class="topbar-title">SkriningGizi · Prediksi <em>Stunting</em> Balita</div>
        <div class="topbar-sub">
            WHO 2006 · Permenkes No. 2 Tahun 2020 · Ensemble CatBoost + XGBoost · 70+ Fitur
        </div>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">AI Screening System</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PREDIKSI & ANALISIS (ENHANCED)
# ═══════════════════════════════════════════════════════════════
if menu.startswith("🏠"):
    st.markdown('<p class="slabel">Data Antropometri Anak</p>', unsafe_allow_html=True)

    # ── INPUT FORM
    col1, col2, col3 = st.columns(3)
    with col1:
        umur_bulan    = st.number_input("Umur (Bulan)", 0, 60, 24, 1)
        berat         = st.number_input("Berat Badan (kg)", 1.0, 35.0, 12.0, 0.1, format="%.1f")
    with col2:
        tinggi        = st.number_input("Tinggi / Panjang Badan (cm)", 40.0, 130.0, 87.0, 0.1, format="%.1f")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"])
    with col3:
        cara_ukur     = st.selectbox("Cara Pengukuran",
                                     ["Terlentang — Panjang Badan", "Berdiri — Tinggi Badan"],
                                     help="Terlentang < 24 bln · Berdiri ≥ 24 bln")
        nama_anak     = st.text_input("Nama Anak (opsional)", placeholder="contoh: Budi")

    # ── KOREKSI PREMATUR
    with st.expander("🔧 Koreksi Usia Prematur (opsional)"):
        prematur       = st.checkbox("Anak lahir prematur (< 37 minggu)")
        usia_gestasi   = st.number_input("Usia gestasi saat lahir (minggu)", 24, 36, 32, 1,
                                          disabled=not prematur)
        if prematur:
            koreksi_bln = round((40 - usia_gestasi) / 4.33, 1)
            umur_terkoreksi = max(0, umur_bulan - koreksi_bln)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.6rem;margin-top:.3rem;">
                <span class="prem-tag">🔧 Koreksi: −{koreksi_bln:.1f} bln
                → Usia terkoreksi: <b>{umur_terkoreksi:.1f} bln</b></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            umur_terkoreksi = umur_bulan

    cu_mode = "Berdiri" if cara_ukur.startswith("Berdiri") else "Terlentang"
    jk      = "L" if jenis_kelamin=="Laki-laki" else "P"

    if umur_terkoreksi < 24 and cu_mode=="Berdiri":
        st.warning("Untuk usia < 24 bulan, disarankan pengukuran Terlentang (panjang badan).")
    elif umur_terkoreksi >= 24 and cu_mode=="Terlentang":
        st.warning("Untuk usia ≥ 24 bulan, disarankan pengukuran Berdiri (tinggi badan).")

    for w in validate_inputs(umur_terkoreksi, berat, tinggi, jk):
        st.warning(w)

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analisis & Prediksi Sekarang", key="predict_main")

    if predict_btn:
        with st.spinner("Menganalisis data antropometri..."):
            try:
                model, threshold, _ = load_model()
                X_input, zs_bbu, zs_bbtb = build_features(
                    umur_terkoreksi, jk, berat, tinggi, cu_mode)
                proba         = model.predict_proba(X_input)[0]
                prob_stunting = proba[1]
                pct           = prob_stunting * 100
                confidence    = max(proba) * 100
                # TB/U z-score & percentile
                zs_tbu        = zscore_tbu(tinggi, umur_terkoreksi, jk)
                pct_bbu       = zscore_to_percentile(zs_bbu)
                pct_bbtb      = zscore_to_percentile(zs_bbtb)
                pct_tbu       = zscore_to_percentile(zs_tbu)
                med_tbu_val   = median_tbu(umur_terkoreksi, jk)
            except Exception as e:
                st.error(f"Error saat prediksi: {e}"); st.stop()

        tier = risk_tier(pct)
        st.session_state.last_result = {
            "umur": umur_terkoreksi, "umur_raw": umur_bulan,
            "berat": berat, "tinggi": tinggi,
            "jk": jenis_kelamin, "nama": nama_anak,
            "pct": pct, "zs_bbu": zs_bbu, "zs_bbtb": zs_bbtb,
            "zs_tbu": zs_tbu, "tier": tier, "confidence": confidence,
            "cara": cu_mode, "pct_tbu": pct_tbu,
        }

        # Save to history
        st.session_state.history.append({
            "Waktu": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Nama": nama_anak or "—", "Umur (bln)": umur_terkoreksi,
            "BB (kg)": berat, "TB (cm)": tinggi, "JK": jenis_kelamin,
            "ZS BB/U": round(zs_bbu,2), "ZS BB/TB": round(zs_bbtb,2),
            "ZS TB/U": round(zs_tbu,2),
            "Prob (%)": round(pct,1), "Hasil": tier["title"],
        })

        # Save to tracking if nama provided
        if nama_anak:
            if nama_anak not in st.session_state.tracking:
                st.session_state.tracking[nama_anak] = []
            st.session_state.tracking[nama_anak].append({
                "tanggal": datetime.date.today().isoformat(),
                "umur": umur_terkoreksi, "berat": berat, "tinggi": tinggi,
                "zs_bbu": zs_bbu, "zs_bbtb": zs_bbtb, "zs_tbu": zs_tbu,
                "prob": pct,
            })

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="slabel">Hasil Analisis AI</p>', unsafe_allow_html=True)

        # ── RESULT PANEL
        bar_pct = max(2, min(97, pct))
        st.markdown(f"""
        <div class="rp {tier['cls']}">
            <div class="rp-eyebrow" style="color:{tier['color']};">{tier['eyebrow']}</div>
            <div class="rp-icon">{tier['icon']}</div>
            <div class="rp-headline" style="color:{tier['color']};">{tier['title']}</div>
            <div class="rp-prob">Probabilitas stunting berdasarkan model AI &nbsp;—&nbsp;
                <strong style="color:{tier['color']};font-size:1rem;">{pct:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE
        conf_col = "#34d399" if confidence>=80 else ("#facc15" if confidence>=60 else "#fb923c")
        conf_lbl = "Tinggi" if confidence>=80 else ("Sedang" if confidence>=60 else "Rendah")
        st.markdown(f"""
        <div style="text-align:center;margin:.6rem 0 1rem;">
            <span class="conf-badge"
                style="background:rgba(201,168,76,.1);border-color:var(--border-gold);color:var(--gold-lt);">
                🧠 Kepercayaan Model:
                <strong style="color:{conf_col};">{confidence:.1f}%</strong>
                &nbsp;<span style="color:{conf_col};font-size:.73rem;">(Keyakinan {conf_lbl})</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([6, 5])

        with r1:
            # ── THREE Z-SCORE CARDS (BB/U, BB/TB, TB/U)
            bc,  bcol,  bbg  = bbu_style(zs_bbu)
            btbc,btbcol,btbbg= bbtb_style(zs_bbtb)
            tc,  tcol,  tbg, teng = tbu_style(zs_tbu)
            st.markdown(f"""
            <div class="zsgrid3">
                <div class="zscard">
                    <div class="zstag">BB / Umur</div>
                    <div class="zsval" style="color:{bcol};">{zs_bbu:+.2f}</div>
                    <div class="zsdesc">Berat Badan per Usia</div>
                    <span class="zsbadge" style="background:{bbg};color:{bcol};">{bc}</span>
                    <div class="zspct">Persentil ke-{pct_bbu:.0f}</div>
                </div>
                <div class="zscard">
                    <div class="zstag">BB / Tinggi</div>
                    <div class="zsval" style="color:{btbcol};">{zs_bbtb:+.2f}</div>
                    <div class="zsdesc">Wasting / Status Gizi</div>
                    <span class="zsbadge" style="background:{btbbg};color:{btbcol};">{btbc}</span>
                    <div class="zspct">Persentil ke-{pct_bbtb:.0f}</div>
                </div>
                <div class="zscard" style="border-color:{'rgba(248,113,113,.3)' if zs_tbu < -2 else 'var(--border)'};">
                    <div class="zstag">TB / Umur ★</div>
                    <div class="zsval" style="color:{tcol};">{zs_tbu:+.2f}</div>
                    <div class="zsdesc">Indikator Stunting</div>
                    <span class="zsbadge" style="background:{tbg};color:{tcol};">{tc}</span>
                    <div class="zspct">Persentil ke-{pct_tbu:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── GAUGE
            st.markdown(f"""
            <div class="gwrap">
                <div class="gtop">
                    <span class="gttl">Probabilitas Stunting (AI Model)</span>
                    <span class="gnum" style="color:{tier['color']};">{pct:.1f}<span style="font-size:.85rem;opacity:.55;">%</span></span>
                </div>
                <div class="gtrack">
                    <div class="gbar" style="width:100%;"></div>
                    <div class="gpip" style="left:{bar_pct}%;color:{tier['color']};background:{tier['color']};"></div>
                </div>
                <div class="gtick"><span>Normal</span><span>Batas 45%</span><span>Risiko Tinggi</span></div>
            </div>
            """, unsafe_allow_html=True)

            # ── THREE-INDEX STATUS TABLE (Permenkes)
            st.markdown('<div class="slabel" style="margin-top:.8rem;">Status Gizi Lengkap — Permenkes No.2/2020</div>', unsafe_allow_html=True)
            def _row_cls(cat): return 'stunting-row' if cat in ["Pendek","Sangat Pendek"] else ''
            tbu_cat, tbu_col, _, _ = tbu_style(zs_tbu)
            bbu_cat, bbu_col, _, = bbu_style(zs_bbu)
            bbtb_cat, bbtb_col, _ = bbtb_style(zs_bbtb)
            st.markdown(f"""
            <table class="tistable">
                <thead><tr>
                    <th>Indeks</th><th>Z-Score</th><th>Persentil</th>
                    <th>Median Ref.</th><th>Kategori</th>
                </tr></thead>
                <tbody>
                    <tr class="{_row_cls(tbu_cat)}">
                        <td class="key-index">TB/U ★ Stunting</td>
                        <td style="color:{tbu_col};font-family:'Cormorant Garamond',serif;font-size:1.1rem;">{zs_tbu:+.2f}</td>
                        <td>P{pct_tbu:.0f}</td>
                        <td>{med_tbu_val} cm</td>
                        <td><span style="color:{tbu_col};font-weight:500;">{tbu_cat}</span></td>
                    </tr>
                    <tr>
                        <td>BB/U Underweight</td>
                        <td style="color:{bbu_col};font-family:'Cormorant Garamond',serif;font-size:1.1rem;">{zs_bbu:+.2f}</td>
                        <td>P{pct_bbu:.0f}</td>
                        <td>—</td>
                        <td><span style="color:{bbu_col};font-weight:500;">{bbu_cat}</span></td>
                    </tr>
                    <tr>
                        <td>BB/TB Wasting</td>
                        <td style="color:{bbtb_col};font-family:'Cormorant Garamond',serif;font-size:1.1rem;">{zs_bbtb:+.2f}</td>
                        <td>P{pct_bbtb:.0f}</td>
                        <td>—</td>
                        <td><span style="color:{bbtb_col};font-weight:500;">{bbtb_cat}</span></td>
                    </tr>
                </tbody>
            </table>
            <div style="font-size:.71rem;color:#475569;margin-top:.35rem;">★ TB/U adalah indikator utama stunting per WHO & Permenkes.</div>
            """, unsafe_allow_html=True)

        with r2:
            # ── INTERPRETASI
            bands = [
                ("<b>0 – 44%</b>",  "Normal / Aman",          "#34d399", pct < 45),
                ("<b>45 – 64%</b>", "Perlu Diwaspadai",        "#facc15", 45<=pct<65),
                ("<b>65 – 79%</b>", "Risiko Stunting Sedang",  "#fb923c", 65<=pct<80),
                ("<b>≥ 80%</b>",    "Risiko Stunting Tinggi",  "#f87171", pct>=80),
            ]
            bands_html = ""
            for rng, lbl, col, is_active in bands:
                active_style = f"background:{col}18;border-color:{col}40;" if is_active else "border-color:transparent;"
                arrow = f"<span style='color:{col};'>◀</span>" if is_active else ""
                bands_html += f"""
                <div class="irow {'active' if is_active else ''}" style="{active_style}">
                    <div class="irow-dot" style="background:{col};{'box-shadow:0 0 5px '+col if is_active else ''}"></div>
                    <span style="color:{'#f1f5f9' if is_active else '#64748b'};flex:1;">{rng} — {lbl}</span>
                    {arrow}
                </div>"""
            st.markdown(f"""
            <div class="ibox">
                <div class="ibox-title">📊 Interpretasi Probabilitas AI</div>
                <div style="font-size:.78rem;color:#94a3b8;margin-bottom:.7rem;">
                    Probabilitas: <strong style="color:{tier['color']};">{pct:.1f}%</strong>
                    &nbsp;·&nbsp; Level: <strong style="color:{tier['color']};">{tier['level']}</strong>
                </div>
                {bands_html}
                <div style="font-size:.7rem;color:#475569;margin-top:.6rem;font-style:italic;">
                    Ambang batas dari optimasi F1-Score. Bukan pengganti diagnosis medis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── META PILLS
            bmi_val  = berat/(tinggi/100)**2
            kel_map  = {0:"ASI Eksklusif",1:"MPASI Awal",2:"Baduta / 1000 HPK",3:"Batita",4:"Balita"}
            bins_k   = [-1,6,11,23,36,60]; kel = 0
            for i in range(len(bins_k)-1):
                if umur_terkoreksi>bins_k[i] and umur_terkoreksi<=bins_k[i+1]: kel=[0,1,2,3,4][i]
            hpk_col  = "#34d399" if umur_terkoreksi<=23 else "#64748b"
            hpk_text = "Aktif" if umur_terkoreksi<=23 else "Lewat"
            selisih_med = tinggi - med_tbu_val
            sel_col  = "#34d399" if selisih_med >= 0 else "#fb923c"
            st.markdown(f"""
            <div class="metarow">
                <div class="mpill">
                    <div class="mplabel">BMI Anak</div>
                    <div class="mpval">{bmi_val:.1f}</div>
                    <div class="mpsub">kg/m²</div>
                </div>
                <div class="mpill">
                    <div class="mplabel">Window 1000 HPK</div>
                    <div class="mpval" style="color:{hpk_col};font-size:1rem;">{hpk_text}</div>
                    <div class="mpsub">{kel_map.get(kel,'—')}</div>
                </div>
            </div>
            <div class="metarow">
                <div class="mpill">
                    <div class="mplabel">Selisih dari Median TB</div>
                    <div class="mpval" style="color:{sel_col};">{selisih_med:+.1f}</div>
                    <div class="mpsub">vs. {med_tbu_val} cm (WHO)</div>
                </div>
                <div class="mpill">
                    <div class="mplabel">Kepercayaan AI</div>
                    <div class="mpval" style="color:{conf_col};">{confidence:.0f}%</div>
                    <div class="mpsub">{conf_lbl}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── REKOMENDASI
            border_col = tier["color"]
            recs_html  = "".join([f'<div class="rec-item">{r}</div>' for r in tier["recs"]])
            st.markdown(f"""
            <div class="recbox" style="border-left-color:{border_col};">
                <span class="acc" style="color:{border_col};">Rekomendasi Intervensi</span>
                <div style="margin-top:.5rem;">{recs_html}</div>
                <div class="disc">
                    Hasil skrining AI — tidak menggantikan diagnosis klinis profesional.
                    Konsultasikan ke tenaga kesehatan untuk penilaian menyeluruh.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── TAMBAH KE TRACKING NOTIF
        if nama_anak:
            st.success(f"✅ Data {nama_anak} disimpan ke Rekam Tumbuh Kembang. Lihat di menu 📈.")
        else:
            st.info("💡 Isi nama anak untuk menyimpan data ke fitur Rekam Tumbuh Kembang.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — REKAM TUMBUH KEMBANG (NEW FEATURE)
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📈"):
    st.markdown('<p class="slabel">Rekam Tumbuh Kembang — Pantau Perkembangan dari Waktu ke Waktu</p>',
                unsafe_allow_html=True)

    tracking = st.session_state.tracking
    names    = list(tracking.keys())

    # ── INPUT MANUAL REKAM
    with st.expander("➕  Tambah Catatan Pengukuran Manual", expanded=not bool(names)):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_nama  = st.text_input("Nama Anak", key="t_nama", placeholder="Nama anak")
            t_umur  = st.number_input("Umur (Bulan)", 0, 60, 12, 1, key="t_umur")
        with c2:
            t_berat  = st.number_input("BB (kg)", 1.0, 35.0, 9.0, .1, format="%.1f", key="t_berat")
            t_tinggi = st.number_input("TB (cm)", 40.0, 130.0, 75.0, .1, format="%.1f", key="t_tinggi")
        with c3:
            t_jk   = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"], key="t_jk")
            t_cara = st.selectbox("Cara Ukur", ["Terlentang — Panjang Badan","Berdiri — Tinggi Badan"], key="t_cara")

        if st.button("Tambah ke Rekam", key="add_track"):
            if not t_nama.strip():
                st.warning("Masukkan nama anak terlebih dahulu.")
            else:
                try:
                    model, threshold, _ = load_model()
                    t_jk_c  = "L" if t_jk=="Laki-laki" else "P"
                    t_cu    = "Berdiri" if t_cara.startswith("Berdiri") else "Terlentang"
                    X, zb, zbt = build_features(t_umur, t_jk_c, t_berat, t_tinggi, t_cu)
                    prob = model.predict_proba(X)[0][1]*100
                    zt   = zscore_tbu(t_tinggi, t_umur, t_jk_c)
                    if t_nama not in tracking:
                        tracking[t_nama] = []
                    tracking[t_nama].append({
                        "tanggal": datetime.date.today().isoformat(),
                        "umur": t_umur, "berat": t_berat, "tinggi": t_tinggi,
                        "zs_bbu": zb, "zs_bbtb": zbt, "zs_tbu": zt, "prob": prob,
                        "jk": t_jk
                    })
                    st.session_state.tracking = tracking
                    st.success(f"✅ Data {t_nama} berhasil ditambahkan!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    if not names:
        st.info("Belum ada rekaman pertumbuhan. Lakukan prediksi (isi nama anak) atau tambah manual di atas.")
        st.stop()

    # ── SELECT ANAK
    selected_child = st.selectbox("Pilih Anak", names, key="sel_child")
    records = tracking[selected_child]
    records_sorted = sorted(records, key=lambda r: r["umur"])

    # ── SUMMARY METRICS
    first_r = records_sorted[0]; last_r = records_sorted[-1]
    delta_bb  = last_r["berat"]  - first_r["berat"]
    delta_tb  = last_r["tinggi"] - first_r["tinggi"]
    delta_prob= last_r["prob"]   - first_r["prob"]
    delta_tbu = last_r["zs_tbu"] - first_r["zs_tbu"]

    def _delta_col(v, positive_good=True):
        if v == 0: return "#94a3b8"
        good = v > 0 if positive_good else v < 0
        return "#34d399" if good else "#f87171"

    st.markdown(f"""
    <div class="metarow">
        <div class="mpill">
            <div class="mplabel">Jumlah Pengukuran</div>
            <div class="mpval" style="color:#e8d48b;">{len(records)}</div>
            <div class="mpsub">titik data</div>
        </div>
        <div class="mpill">
            <div class="mplabel">Δ Berat Badan</div>
            <div class="mpval" style="color:{_delta_col(delta_bb)};">{delta_bb:+.1f}</div>
            <div class="mpsub">kg sejak awal</div>
        </div>
        <div class="mpill">
            <div class="mplabel">Δ Tinggi Badan</div>
            <div class="mpval" style="color:{_delta_col(delta_tb)};">{delta_tb:+.1f}</div>
            <div class="mpsub">cm sejak awal</div>
        </div>
        <div class="mpill">
            <div class="mplabel">Δ Z-Score TB/U</div>
            <div class="mpval" style="color:{_delta_col(delta_tbu)};">{delta_tbu:+.2f}</div>
            <div class="mpsub">perubahan stunting</div>
        </div>
        <div class="mpill">
            <div class="mplabel">Δ Probabilitas</div>
            <div class="mpval" style="color:{_delta_col(delta_prob, positive_good=False)};">{delta_prob:+.1f}%</div>
            <div class="mpsub">risiko stunting</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TREND CHART
    if len(records_sorted) >= 2:
        fig_track = plot_tracking_chart(records_sorted, last_r.get("jk","L")[:1])
        if fig_track:
            st.pyplot(fig_track, use_container_width=True)
            plt.close(fig_track)
    else:
        st.info("Tambahkan minimal 2 pengukuran untuk menampilkan grafik tren.")

    # ── RECORDS TABLE
    st.markdown('<p class="slabel" style="margin-top:.8rem;">Detail Rekaman</p>', unsafe_allow_html=True)
    rows_html = ""
    for rec in records_sorted:
        tc, tcol, _, _ = tbu_style(rec["zs_tbu"])
        pcolor = "#34d399" if rec["prob"] < 45 else ("#facc15" if rec["prob"] < 65 else ("#fb923c" if rec["prob"] < 80 else "#f87171"))
        rows_html += f"""
        <tr>
            <td>{rec.get('tanggal','—')}</td>
            <td style="text-align:center;">{rec['umur']} bln</td>
            <td style="text-align:center;">{rec['berat']:.1f}</td>
            <td style="text-align:center;">{rec['tinggi']:.1f}</td>
            <td style="text-align:center;font-family:'Cormorant Garamond',serif;font-size:1.05rem;color:{tcol};">{rec['zs_tbu']:+.2f}</td>
            <td style="text-align:center;"><span style="color:{tcol};font-size:.8rem;">{tc}</span></td>
            <td style="text-align:center;color:{pcolor};font-weight:500;">{rec['prob']:.1f}%</td>
        </tr>"""
    st.markdown(f"""
    <div class="hist-wrap">
    <table>
        <thead><tr>
            <th>Tanggal</th><th>Umur</th><th>BB (kg)</th><th>TB (cm)</th>
            <th>ZS TB/U</th><th>Status Tinggi</th><th>Prob Stunting</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ── HAPUS DATA ANAK
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3,1])
    with col_b:
        if st.button(f"🗑️  Hapus Data {selected_child}", key="del_child"):
            del st.session_state.tracking[selected_child]
            st.rerun()

    # ── EXPORT
    df_track = pd.DataFrame(records_sorted)
    csv_track = df_track.to_csv(index=False).encode("utf-8")
    with col_a:
        st.download_button(
            f"⬇️  Unduh Rekaman {selected_child} (CSV)",
            data=csv_track,
            file_name=f"tumbuh_kembang_{selected_child}_{datetime.date.today()}.csv",
            mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — GRAFIK PERTUMBUHAN WHO
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📊"):
    st.markdown('<p class="slabel">WHO Growth Chart — Kurva Pertumbuhan Standar</p>', unsafe_allow_html=True)

    res = st.session_state.last_result
    if res:
        jk_code = "L" if res["jk"]=="Laki-laki" else "P"
        st.info(f"📌 Data terakhir: **{res['nama'] or 'Anak'}** — "
                f"Usia {res['umur']} bln · BB {res['berat']} kg · TB {res['tinggi']} cm")
        umur_g=res["umur"]; berat_g=res["berat"]; tinggi_g=res["tinggi"]; jk_g=jk_code
    else:
        st.info("Belum ada prediksi. Isi form di bawah untuk plot kurva.")
        c1,c2,c3 = st.columns(3)
        with c1: umur_g  = st.number_input("Umur (bln)", 0,60,24,1,key="g_umur")
        with c2: berat_g = st.number_input("BB (kg)",1.0,35.0,12.0,.1,format="%.1f",key="g_berat")
        with c3: tinggi_g= st.number_input("TB (cm)",40.0,130.0,87.0,.1,format="%.1f",key="g_tinggi")
        jk_g = "L" if st.selectbox("JK",["Laki-laki","Perempuan"],key="g_jk")=="Laki-laki" else "P"

    fig = plot_growth_chart(umur_g, tinggi_g, berat_g, jk_g)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── TB/U interpretation
    zs_tbu_g = zscore_tbu(tinggi_g, umur_g, jk_g)
    med_g     = median_tbu(umur_g, jk_g)
    tc_g, tcol_g, _, teng_g = tbu_style(zs_tbu_g)
    pct_g = zscore_to_percentile(zs_tbu_g)

    status_msg = {
        "Sangat Pendek": "🔴 Tinggi badan berada di bawah −3 SD — Sangat Pendek (Severely Stunted)",
        "Pendek":        "🟠 Tinggi badan berada di bawah −2 SD — Pendek (Stunted)",
        "Normal":        "✅ Tinggi badan berada dalam rentang normal WHO",
        "Tinggi":        "🔵 Tinggi badan berada di atas +3 SD",
    }
    st.markdown(f"""
    <div class="ibox" style="margin-top:1rem;">
        <div class="ibox-title">📊 Interpretasi TB/U pada Kurva WHO</div>
        <div style="font-size:.88rem;color:#94a3b8;line-height:1.8;">
            <b style="color:#e8d48b;">Tinggi badan anak:</b> {tinggi_g:.1f} cm &nbsp;·&nbsp;
            <b style="color:#94a3b8;">Median WHO usia {umur_g} bln:</b> {med_g} cm &nbsp;·&nbsp;
            <b style="color:#94a3b8;">Z-Score TB/U:</b>
            <span style="color:{tcol_g};font-weight:600;">{zs_tbu_g:+.2f} ({tc_g})</span>
            &nbsp;·&nbsp; Persentil ke-<b style="color:{tcol_g};">{pct_g:.0f}</b>
        </div>
        <div style="font-size:.85rem;color:#94a3b8;margin-top:.4rem;">
            {status_msg.get(tc_g, "")}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ANALISIS FITUR
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
            Nilai dihitung dari rata-rata gain ternormalisasi.<br><br>
            <b style="color:#e8d48b;">Temuan utama:</b>
            Indeks gizi berbasis Z-Score (BB/TB & BB/U) mendominasi prediksi karena langsung
            mencerminkan status gizi anak relatif terhadap standar WHO. Tinggi badan raw
            berkontribusi signifikan karena menjadi pembagi dalam indeks BB/TB.
            Interaksi umur × z-score menangkap perbedaan risiko antar kelompok usia kritis (1000 HPK).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="slabel" style="margin-top:1.2rem;">Tabel Kontribusi Fitur</p>', unsafe_allow_html=True)
    df_fi = pd.DataFrame(FEATURE_IMPORTANCE, columns=["Fitur","Kontribusi (%)"])
    df_fi.index = df_fi.index + 1
    st.dataframe(
        df_fi.style.background_gradient(subset=["Kontribusi (%)"], cmap="YlOrRd"),
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — SIMULASI INTERVENSI
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("⚡"):
    st.markdown('<p class="slabel">Simulasi Intervensi — Bagaimana Jika Parameter Berubah?</p>',
                unsafe_allow_html=True)

    res = st.session_state.last_result
    if not res:
        st.warning("⚠️ Lakukan prediksi tunggal terlebih dahulu di menu Prediksi & Analisis.")
        st.stop()

    base_berat=res["berat"]; base_tinggi=res["tinggi"]
    base_umur=res["umur"]; jk="L" if res["jk"]=="Laki-laki" else "P"
    cara=res["cara"]; base_pct=res["pct"]

    st.markdown(f"""
    <div class="gcard" style="margin-bottom:1.2rem;">
        <div class="ibox-title">📌 Data Dasar Prediksi Terakhir</div>
        <div style="font-size:.88rem;color:#94a3b8;">
            Anak: <b style="color:#e8d48b;">{res['nama'] or '—'}</b> &nbsp;·&nbsp;
            Usia: <b style="color:#e8d48b;">{base_umur} bln</b> &nbsp;·&nbsp;
            BB: <b style="color:#e8d48b;">{base_berat} kg</b> &nbsp;·&nbsp;
            TB: <b style="color:#e8d48b;">{base_tinggi} cm</b> &nbsp;·&nbsp;
            Prob dasar: <b style="color:{res['tier']['color']};">{base_pct:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        delta_bb  = st.slider("Δ Berat Badan (kg)",   -3.0, 5.0, 0.0, 0.1, format="%+.1f kg")
        delta_tb  = st.slider("Δ Tinggi Badan (cm)",  -5.0, 10.0, 0.0, 0.5, format="%+.1f cm")
    with c2:
        delta_umur= st.slider("Proyeksi Usia Ke Depan (+bln)", 0, 12, 0, 1, format="+%d bln")

    with st.spinner("Menghitung simulasi..."):
        try:
            model, threshold, _ = load_model()
            new_berat  = max(1.0, base_berat + delta_bb)
            new_tinggi = max(40.0, base_tinggi + delta_tb)
            new_umur   = min(60, base_umur + delta_umur)
            X_new, zs_new_bbu, zs_new_bbtb = build_features(new_umur, jk, new_berat, new_tinggi, cara)
            pct_new    = model.predict_proba(X_new)[0][1]*100
            zs_new_tbu = zscore_tbu(new_tinggi, new_umur, jk)
            tier_new   = risk_tier(pct_new)
            delta_pct  = pct_new - base_pct
            pct_new_tbu= zscore_to_percentile(zs_new_tbu)

            # Sensitivity curve
            bb_range = np.arange(max(1.0, base_berat-3), min(30, base_berat+5.1), 0.4)
            pct_bb   = [model.predict_proba(build_features(new_umur,jk,bb,new_tinggi,cara)[0])[0][1]*100
                        for bb in bb_range]
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

    arrow  = "↓" if delta_pct < 0 else "↑"
    arr_col= "#34d399" if delta_pct < 0 else "#f87171"
    st.markdown(f"""
    <div class="rp {tier_new['cls']}" style="margin-top:1rem;">
        <div class="rp-eyebrow" style="color:{tier_new['color']};">Hasil Simulasi</div>
        <div class="rp-icon">{tier_new['icon']}</div>
        <div class="rp-headline" style="color:{tier_new['color']};">{tier_new['title']}</div>
        <div class="rp-prob">
            Probabilitas: <strong style="color:{tier_new['color']};">{pct_new:.1f}%</strong>
            &nbsp;<span style="color:{arr_col};font-weight:600;">{arrow} {abs(delta_pct):.1f}%</span>
            dari data dasar ({base_pct:.1f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Z-score comparison
    bc_n,bcol_n,bbg_n    = bbu_style(zs_new_bbu)
    btbc_n,btbcol_n,btbbg_n = bbtb_style(zs_new_bbtb)
    tc_n, tcol_n, tbg_n, _ = tbu_style(zs_new_tbu)
    c_a, c_b, c_c = st.columns(3)
    for col_ui, tag, z, cat, zcol, zbg in [
        (c_a,"BB/U (Sim.)",  zs_new_bbu,  bc_n,  bcol_n,  bbg_n),
        (c_b,"BB/TB (Sim.)", zs_new_bbtb, btbc_n,btbcol_n,btbbg_n),
        (c_c,"TB/U ★ (Sim.)",zs_new_tbu,  tc_n,  tcol_n,  tbg_n),
    ]:
        col_ui.markdown(f"""
        <div class="zscard" style="text-align:center;padding:1rem;">
            <div class="zstag">{tag}</div>
            <div class="zsval" style="color:{zcol};">{z:+.2f}</div>
            <div class="zspct">P{zscore_to_percentile(z):.0f}</div>
            <span class="zsbadge" style="background:{zbg};color:{zcol};">{cat}</span>
        </div>""", unsafe_allow_html=True)

    # Sensitivity chart
    st.markdown('<p class="slabel" style="margin-top:1.2rem;">Kurva Sensitivitas — Probabilitas vs Berat Badan</p>',
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10,3.6))
    fig2.patch.set_facecolor("#1a2535"); ax2.set_facecolor("#111827")
    ax2.fill_between(bb_range, 45, 100, color="#f87171", alpha=.06)
    ax2.axhline(45, color="#facc15", lw=1, linestyle="--", alpha=.6, label="Batas 45%")
    ax2.plot(bb_range, pct_bb, color="#4fd1c5", lw=2.3)
    ax2.scatter([new_berat], [pct_new], color="#c9a84c", s=80, zorder=5, label=f"Simulasi ({new_berat:.1f} kg)")
    ax2.scatter([base_berat],[base_pct], color="#94a3b8", s=60, zorder=4, label=f"Dasar ({base_berat:.1f} kg)")
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

    # TB sensitivity
    tb_range = np.arange(max(40.0, base_tinggi-5), min(130.0, base_tinggi+10.1), 0.5)
    pct_tb   = [model.predict_proba(build_features(new_umur,jk,new_berat,tb,cara)[0])[0][1]*100
                for tb in tb_range]
    st.markdown('<p class="slabel">Kurva Sensitivitas — Probabilitas vs Tinggi Badan</p>',
                unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10,3.4))
    fig3.patch.set_facecolor("#1a2535"); ax3.set_facecolor("#111827")
    ax3.fill_between(tb_range, 45, 100, color="#f87171", alpha=.06)
    ax3.axhline(45, color="#facc15", lw=1, linestyle="--", alpha=.6, label="Batas 45%")
    ax3.plot(tb_range, pct_tb, color="#a78bfa", lw=2.3)
    ax3.scatter([new_tinggi],[pct_new], color="#c9a84c", s=80, zorder=5, label=f"Simulasi ({new_tinggi:.1f} cm)")
    ax3.set_xlabel("Tinggi Badan (cm)", color="#64748b", fontsize=9)
    ax3.set_ylabel("Probabilitas Stunting (%)", color="#64748b", fontsize=9)
    ax3.set_title("Sensitivitas Probabilitas Stunting terhadap Perubahan Tinggi Badan",
                  color="#e8d48b", fontsize=10, fontweight='normal')
    ax3.tick_params(colors="#64748b", labelsize=8.5)
    for sp in ['bottom','left']: ax3.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax3.spines[sp].set_visible(False)
    ax3.grid(color="#1f2d42", linestyle="--", linewidth=0.7, alpha=0.7)
    ax3.legend(fontsize=8, facecolor="#1a2535", edgecolor="#1f2d42", labelcolor="#94a3b8")
    ax3.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — RIWAYAT PREDIKSI
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📋"):
    st.markdown('<p class="slabel">Riwayat Prediksi — Sesi Ini</p>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info("Belum ada riwayat. Lakukan prediksi di menu Prediksi & Analisis.")
    else:
        df_hist = pd.DataFrame(hist)
        total   = len(df_hist)
        n_normal= sum("Normal" in r for r in df_hist["Hasil"])
        avg_prob= df_hist["Prob (%)"].mean()
        n_stunt_direct = sum(r < -2 for r in df_hist.get("ZS TB/U", pd.Series(dtype=float)))

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val,clr in [
            (c1,"Total Prediksi",str(total),"#e8d48b"),
            (c2,"Status Normal",str(n_normal),"#34d399"),
            (c3,"Indikasi Risiko",str(total-n_normal),"#f87171"),
            (c4,"Rata-rata Prob",f"{avg_prob:.1f}%","#4fd1c5"),
        ]:
            col.markdown(f"""
            <div style="background:var(--navy-3);border:1px solid var(--border);border-radius:12px;
                        padding:1rem;text-align:center;margin-bottom:.6rem;">
                <div style="font-family:'Cormorant Garamond',serif;font-size:1.8rem;color:{clr};">{val}</div>
                <div style="font-size:.68rem;color:var(--text-3);text-transform:uppercase;letter-spacing:.1em;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        # Table
        rows_html = ""
        for _, row in df_hist.iterrows():
            rst = row["Hasil"]
            if "Normal" in rst:       bs = "color:#34d399;background:rgba(52,211,153,.10);padding:.18rem .55rem;border-radius:5px;"
            elif "Diwaspadai" in rst: bs = "color:#facc15;background:rgba(250,204,21,.10);padding:.18rem .55rem;border-radius:5px;"
            elif "Sedang" in rst:     bs = "color:#fb923c;background:rgba(251,146,60,.10);padding:.18rem .55rem;border-radius:5px;"
            else:                     bs = "color:#f87171;background:rgba(248,113,113,.10);padding:.18rem .55rem;border-radius:5px;"
            tbu_z   = row.get("ZS TB/U", "—")
            tc_h, tcol_h, _, _ = tbu_style(float(tbu_z)) if tbu_z != "—" else ("—","#64748b","","")
            rows_html += f"""
            <tr>
                <td>{row['Waktu']}</td><td>{row['Nama']}</td>
                <td style="text-align:center;">{row['Umur (bln)']}</td>
                <td style="text-align:center;">{row['BB (kg)']}</td>
                <td style="text-align:center;">{row['TB (cm)']}</td>
                <td style="text-align:center;">{row['JK']}</td>
                <td style="text-align:center;color:{tcol_h};">{tbu_z if tbu_z=='—' else f'{float(tbu_z):+.2f}'}</td>
                <td style="text-align:center;color:{tcol_h};font-size:.79rem;">{tc_h}</td>
                <td style="text-align:center;color:#e8d48b;font-weight:500;">{row['Prob (%)']:.1f}%</td>
                <td><span style="{bs}">{rst}</span></td>
            </tr>"""
        st.markdown(f"""
        <div class="hist-wrap">
        <table><thead><tr>
            <th>Waktu</th><th>Nama</th><th>Umur</th><th>BB</th><th>TB</th>
            <th>JK</th><th>ZS TB/U</th><th>Status TB</th><th>Prob</th><th>Hasil AI</th>
        </tr></thead><tbody>{rows_html}</tbody></table>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        ca, cb = st.columns([3,1])
        with ca:
            csv_data = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Unduh Riwayat CSV", data=csv_data,
                               file_name=f"riwayat_{datetime.date.today()}.csv", mime="text/csv")
        with cb:
            if st.button("🗑️  Hapus Riwayat"):
                st.session_state.history = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — PREDIKSI BATCH EXCEL
# ═══════════════════════════════════════════════════════════════
elif menu.startswith("📁"):
    st.markdown('<p class="slabel">Prediksi Massal — Upload File Excel</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ibox" style="margin-bottom:1.2rem;">
        <div class="ibox-title">📋 Format Kolom File Excel (nama persis, huruf kecil)</div>
        <div style="font-size:.86rem;color:#94a3b8;line-height:1.9;">
            <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">umur_bulan</code>
            <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">jk</code> (L/P)
            <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">berat</code>
            <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">tinggi</code>
            <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">cara_ukur</code>
            (Berdiri/Terlentang)
            <br>Kolom <code style="background:#0b1120;color:#4fd1c5;padding:.12rem .38rem;border-radius:4px;">nama</code> opsional.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Template
    template_df = pd.DataFrame({
        "nama":["Budi","Sari","Andi"], "umur_bulan":[24,36,12], "jk":["L","P","L"],
        "berat":[11.5,13.2,8.8], "tinggi":[85.0,93.5,72.0], "cara_ukur":["Berdiri","Berdiri","Terlentang"]
    })
    tpl_bytes = io.BytesIO()
    template_df.to_excel(tpl_bytes, index=False)
    st.download_button("⬇️  Unduh Template Excel", data=tpl_bytes.getvalue(),
                       file_name="template_prediksi_stunting.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded = st.file_uploader("Upload File Excel (.xlsx / .xls)", type=["xlsx","xls"])
    if uploaded:
        try:
            df_up = pd.read_excel(uploaded)
            df_up.columns = df_up.columns.str.lower().str.strip()
            required = ["umur_bulan","jk","berat","tinggi","cara_ukur"]
            missing  = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"Kolom tidak ditemukan: {missing}"); st.stop()

            model, threshold, _ = load_model()
            results = []
            prog = st.progress(0)
            for idx, row in df_up.iterrows():
                prog.progress((idx+1)/len(df_up))
                try:
                    umur_r = int(row.umur_bulan); jk_r = str(row.jk).strip()
                    bb_r = float(row.berat); tb_r = float(row.tinggi)
                    cu_r = str(row.cara_ukur).strip()
                    X, z_bbu, z_bbtb = build_features(umur_r, jk_r, bb_r, tb_r, cu_r)
                    prob = model.predict_proba(X)[0][1]*100
                    z_tbu= zscore_tbu(tb_r, umur_r, jk_r)
                    tier = risk_tier(prob)
                    bc,_,_  = bbu_style(z_bbu)
                    btbc,_,_= bbtb_style(z_bbtb)
                    tc,tcol,_,teng = tbu_style(z_tbu)
                    results.append({
                        "Nama": row.get("nama","—"), "Umur (bln)": umur_r,
                        "JK": jk_r, "BB (kg)": bb_r, "TB (cm)": tb_r,
                        "ZS BB/U": round(z_bbu,2), "ZS BB/TB": round(z_bbtb,2),
                        "ZS TB/U": round(z_tbu,2),
                        "Status TB/U": tc, "Persentil TB/U": round(zscore_to_percentile(z_tbu),1),
                        "Status BB/U": bc, "Status BB/TB": btbc,
                        "Prob (%)": round(prob,1), "Hasil AI": tier["title"], "Level": tier["level"],
                    })
                except Exception as ex:
                    results.append({"Nama": row.get("nama","?"), "Error": str(ex)})
            prog.empty()

            df_out = pd.DataFrame(results)
            total_b = len(df_out)
            n_stunt_b = sum(df_out.get("Status TB/U","").str.contains("Pendek", na=False))
            n_risk_b  = sum("Normal" not in str(r) for r in df_out.get("Hasil AI",[]))

            st.markdown(f"""
            <div class="metarow" style="margin-top:1rem;">
                <div class="mpill"><div class="mplabel">Total Anak</div>
                    <div class="mpval">{total_b}</div><div class="mpsub">dianalisis</div></div>
                <div class="mpill"><div class="mplabel">Stunting (TB/U)</div>
                    <div class="mpval" style="color:#f87171;">{n_stunt_b}</div>
                    <div class="mpsub">ZS TB/U &lt; −2 SD</div></div>
                <div class="mpill"><div class="mplabel">Risiko AI</div>
                    <div class="mpval" style="color:#fb923c;">{n_risk_b}</div>
                    <div class="mpsub">perlu perhatian</div></div>
                <div class="mpill"><div class="mplabel">Prev. TB/U</div>
                    <div class="mpval" style="color:#facc15;">{n_stunt_b/total_b*100:.1f}%</div>
                    <div class="mpsub">dari total</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(df_out, use_container_width=True)
            csv_out = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Unduh Hasil CSV", data=csv_out,
                               file_name=f"hasil_batch_{datetime.date.today()}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error memproses file: {e}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fnote">
    <span>Permenkes No. 2 Tahun 2020</span> — Standar Antropometri Anak &nbsp;·&nbsp;
    <span>WHO 2006 Multicentre Growth Reference Study</span>
    <br>Model Ensemble CatBoost + XGBoost &nbsp;·&nbsp; Skrining Awal — Bukan Pengganti Diagnosis Medis
</div>
""", unsafe_allow_html=True)
