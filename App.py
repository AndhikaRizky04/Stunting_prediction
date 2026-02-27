import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bisect

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Prediksi Stunting Balita",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# DESIGN SYSTEM — Luxury Dark Medical
# Palette: Deep slate navy + warm gold + soft mint
# Typography: Cormorant Garamond (display) + Outfit (body)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500;600&display=swap');

/* ── TOKENS ── */
:root {
    --navy:       #0b1120;
    --navy-2:     #111827;
    --navy-3:     #1a2535;
    --navy-4:     #1f2d42;
    --glass:      rgba(255,255,255,0.04);
    --glass-b:    rgba(255,255,255,0.07);
    --gold:       #c9a84c;
    --gold-lt:    #e8d48b;
    --gold-dim:   rgba(201,168,76,0.15);
    --mint:       #4fd1c5;
    --danger:     #f87171;
    --danger-dim: rgba(248,113,113,0.12);
    --success:    #34d399;
    --success-dim:rgba(52,211,153,0.12);
    --text-1:     #f1f5f9;
    --text-2:     #94a3b8;
    --text-3:     #64748b;
    --border:     rgba(255,255,255,0.08);
    --border-gold:rgba(201,168,76,0.3);
}

/* ── RESET & BASE ── */
html, body,
[class*="css"],
.stApp,
.main,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"],
div[data-testid="stVerticalBlock"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--text-1) !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    max-width: 720px !important;
    padding: 2rem 1.5rem 5rem !important;
}

/* ── HERO ── */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 3.5rem 2rem 3rem;
    margin-bottom: 2.8rem;
    border-radius: 28px;
    background: var(--navy-3);
    border: 1px solid var(--border-gold);
    overflow: hidden;
    animation: fadeDown 0.7s ease both;
}
.hero-wrap::before {
    content:'';
    position:absolute; inset:0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%,
        rgba(201,168,76,.13) 0%, transparent 70%);
    pointer-events:none;
}
.hero-wrap::after {
    content:'';
    position:absolute; bottom:-1px; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.hero-orb {
    position:absolute;
    width:320px; height:320px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(79,209,197,.06) 0%, transparent 70%);
    top:-120px; right:-80px;
    pointer-events:none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: var(--gold-dim);
    border: 1px solid var(--border-gold);
    color: var(--gold-lt);
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: .35rem .95rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3rem !important;
    font-weight: 300 !important;
    letter-spacing: -.03em !important;
    line-height: 1.1 !important;
    color: var(--text-1) !important;
    margin: 0 0 .4rem !important;
}
.hero-title em {
    font-style: italic;
    color: var(--gold-lt);
}
.hero-divider {
    width: 40px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.1rem auto;
}
.hero-sub {
    font-size: .88rem !important;
    color: var(--text-3) !important;
    font-weight: 300 !important;
    letter-spacing: .03em !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: .65rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
    margin-top: .2rem;
    display: flex;
    align-items: center;
    gap: .7rem;
}
.section-label::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, var(--border-gold), transparent);
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
div[data-testid="stNumberInput"] input {
    background: var(--navy-3) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .95rem !important;
    padding: .65rem .9rem !important;
    transition: border-color .25s, box-shadow .25s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
    outline: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--navy-3) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: 'Outfit', sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
}
div[data-testid="stSelectbox"] ul {
    background: var(--navy-4) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 14px !important;
    padding: .4rem !important;
    box-shadow: 0 16px 48px rgba(0,0,0,.6) !important;
}
div[data-testid="stSelectbox"] li {
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-2) !important;
}
div[data-testid="stSelectbox"] li:hover {
    background: var(--gold-dim) !important;
    color: var(--gold-lt) !important;
}
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p {
    color: var(--text-3) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] button {
    background: var(--navy-4) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-3) !important;
    border-radius: 8px !important;
    transition: color .2s !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: var(--gold) !important;
}

/* ── CTA BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #9a6e1a 0%, #c9a84c 45%, #e8d48b 100%) !important;
    color: #0b1120 !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1.05rem 2rem !important;
    font-size: .95rem !important;
    font-weight: 600 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: 0 8px 30px rgba(201,168,76,.3), inset 0 1px 0 rgba(255,255,255,.2) !important;
    transition: all .3s cubic-bezier(.4,0,.2,1) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 16px 44px rgba(201,168,76,.4), inset 0 1px 0 rgba(255,255,255,.25) !important;
}
.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* ── RESULT PANEL ── */
.result-panel {
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    animation: resultPop 0.65s cubic-bezier(.34,1.56,.64,1) both;
}
.rp-normal {
    background: linear-gradient(160deg, #091a12 0%, #0d2218 100%);
    border: 1px solid rgba(52,211,153,.25);
    box-shadow: 0 0 60px rgba(52,211,153,.06);
}
.rp-normal::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 65% 45% at 50% 0%,
        rgba(52,211,153,.12) 0%, transparent 70%);
}
.rp-waspada {
    background: linear-gradient(160deg, #1a1400 0%, #221a00 100%);
    border: 1px solid rgba(250,204,21,.25);
    box-shadow: 0 0 60px rgba(250,204,21,.05);
}
.rp-waspada::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 65% 45% at 50% 0%,
        rgba(250,204,21,.10) 0%, transparent 70%);
}
.rp-sedang {
    background: linear-gradient(160deg, #1a0e00 0%, #221400 100%);
    border: 1px solid rgba(251,146,60,.25);
    box-shadow: 0 0 60px rgba(251,146,60,.05);
}
.rp-sedang::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 65% 45% at 50% 0%,
        rgba(251,146,60,.11) 0%, transparent 70%);
}
.rp-stunting {
    background: linear-gradient(160deg, #1a0909 0%, #220d0d 100%);
    border: 1px solid rgba(248,113,113,.25);
    box-shadow: 0 0 60px rgba(248,113,113,.06);
}
.rp-stunting::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 65% 45% at 50% 0%,
        rgba(248,113,113,.12) 0%, transparent 70%);
}
.rp-eyebrow {
    font-size: .65rem; font-weight: 600;
    letter-spacing: .2em; text-transform: uppercase;
    opacity: .7; margin-bottom: .9rem;
    position:relative;
}
.rp-icon { font-size: 2.6rem; margin-bottom: .5rem; position:relative; line-height:1; }
.rp-headline {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.5rem; font-weight: 600;
    letter-spacing: -.03em; line-height: 1;
    margin-bottom: .45rem; position:relative;
}
.rp-prob {
    font-size: .85rem; font-weight: 400;
    opacity: .65; position:relative;
}

/* ── Z-SCORE GRID ── */
.zs-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1rem; margin-bottom: 1.2rem;
}
.zs-card {
    background: var(--navy-3);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 1.4rem 1.1rem;
    text-align: center;
    transition: border-color .25s, transform .2s;
}
.zs-card:hover { border-color: var(--border-gold); transform: translateY(-2px); }
.zs-tag {
    font-size: .63rem; letter-spacing: .16em;
    text-transform: uppercase; font-weight: 600;
    color: var(--text-3); margin-bottom: .7rem;
}
.zs-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem; font-weight: 300; line-height: 1;
    margin-bottom: .5rem;
}
.zs-desc { font-size: .7rem; color: var(--text-3); margin-bottom: .55rem; }
.zs-badge {
    font-size: .73rem; font-weight: 500;
    padding: .28rem .75rem; border-radius: 100px;
    display: inline-block;
}

/* ── GAUGE ── */
.gauge-wrap {
    background: var(--navy-3);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
}
.gauge-top {
    display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom: 1rem;
}
.gauge-ttl {
    font-size: .63rem; letter-spacing: .16em;
    text-transform: uppercase; font-weight: 600; color: var(--text-3);
}
.gauge-num {
    font-family:'Cormorant Garamond',serif;
    font-size: 1.7rem; font-weight: 300; line-height: 1;
}
.gauge-track {
    height: 6px; border-radius: 100px;
    background: rgba(255,255,255,.07); position: relative;
}
.gauge-bar {
    position:absolute; left:0; top:0; height:100%;
    border-radius:100px;
    background: linear-gradient(90deg, #34d399 0%, #facc15 55%, #f87171 100%);
}
.gauge-pip {
    position:absolute; top:50%; transform:translate(-50%,-50%);
    width:14px; height:14px; border-radius:50%;
    border: 2px solid var(--navy); z-index:2;
    box-shadow: 0 0 8px currentColor;
}
.gauge-ticks {
    display:flex; justify-content:space-between;
    margin-top:.6rem; font-size:.67rem; color:var(--text-3);
}

/* ── META PILLS ── */
.meta-row {
    display:flex; gap:.9rem; margin-bottom:1.2rem;
}
.meta-pill {
    flex:1; background:var(--navy-3);
    border:1px solid var(--border); border-radius:14px;
    padding:.95rem .7rem; text-align:center;
    transition: border-color .2s;
}
.meta-pill:hover { border-color: var(--border-gold); }
.mp-label { font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; color:var(--text-3); font-weight:600; }
.mp-val {
    font-family:'Cormorant Garamond',serif;
    font-size:1.45rem; font-weight:300; color:var(--text-1);
    margin:.25rem 0 .15rem; line-height:1;
}
.mp-sub { font-size:.67rem; color:var(--text-3); }

/* ── REC STRIP ── */
.rec-strip {
    background: var(--navy-3);
    border:1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 14px; padding: 1.15rem 1.3rem;
    font-size: .87rem; line-height: 1.7;
    color: var(--text-2); margin-bottom: 1rem;
}
.rec-acc { color: var(--gold-lt); font-weight: 500; }
.rec-disc { font-size: .76rem; color: var(--text-3); margin-top: .65rem; font-style: italic; }

/* ── ALERT OVERRIDE ── */
div[data-testid="stAlert"] {
    background: rgba(201,168,76,.08) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 12px !important;
    color: var(--gold-lt) !important;
    font-family:'Outfit',sans-serif !important;
    font-size:.87rem !important;
}

/* ── SPINNER ── */
div[data-testid="stSpinner"] p {
    color: var(--text-2) !important;
    font-family:'Outfit',sans-serif !important;
}

/* ── FOOTER ── */
.footer-note {
    text-align:center; color:var(--text-3);
    font-size:.74rem; line-height:1.9;
    padding-top:1.8rem; margin-top:3rem;
    border-top:1px solid var(--border);
}
.footer-note span { color:var(--text-2); }
.footer-dot { opacity:.4; margin:0 .4rem; }

/* ── ANIMATIONS ── */
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes resultPop {
    from { opacity:0; transform:scale(.93); }
    to   { opacity:1; transform:scale(1); }
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# WHO 2006 GROWTH STANDARDS — LMS Reference Tables
# =====================================================================

WFA_BOYS = [
    [0,0.3487,3.3464,0.14602],[1,0.2297,4.4709,0.13395],[2,0.1970,5.5675,0.12385],
    [3,0.1738,6.3762,0.11578],[4,0.1553,7.0023,0.10943],[5,0.1395,7.5105,0.10452],
    [6,0.1257,7.9340,0.10079],[7,0.1134,8.3030,0.09816],[8,0.1021,8.6420,0.09578],
    [9,0.0917,8.9496,0.09400],[10,0.0820,9.2422,0.09230],[11,0.0730,9.5238,0.09117],
    [12,0.0648,9.7870,0.09026],[13,0.0570,10.0425,0.08957],[14,0.0497,10.2850,0.08921],
    [15,0.0427,10.5244,0.08900],[16,0.0360,10.7540,0.08900],[17,0.0296,10.9762,0.08911],
    [18,0.0234,11.1945,0.08934],[19,0.0173,11.4078,0.08969],[20,0.0115,11.6171,0.09003],
    [21,0.0058,11.8228,0.09062],[22,0.0002,12.0245,0.09112],[23,-0.0054,12.2244,0.09180],
    [24,-0.0108,12.4237,0.09258],[25,-0.0162,12.6166,0.09336],[26,-0.0215,12.8077,0.09418],
    [27,-0.0267,12.9980,0.09508],[28,-0.0318,13.1872,0.09602],[29,-0.0369,13.3754,0.09700],
    [30,-0.0419,13.5638,0.09803],[31,-0.0468,13.7493,0.09907],[32,-0.0517,13.9333,0.10013],
    [33,-0.0565,14.1165,0.10122],[34,-0.0612,14.2990,0.10233],[35,-0.0659,14.4803,0.10344],
    [36,-0.0705,14.6612,0.10456],[37,-0.0750,14.8406,0.10571],[38,-0.0795,15.0189,0.10686],
    [39,-0.0839,15.1966,0.10802],[40,-0.0883,15.3729,0.10918],[41,-0.0926,15.5486,0.11034],
    [42,-0.0969,15.7243,0.11151],[43,-0.1011,15.8999,0.11269],[44,-0.1052,16.0753,0.11385],
    [45,-0.1093,16.2514,0.11503],[46,-0.1134,16.4281,0.11621],[47,-0.1174,16.6055,0.11739],
    [48,-0.1213,16.7836,0.11856],[49,-0.1252,16.9625,0.11973],[50,-0.1290,17.1426,0.12089],
    [51,-0.1328,17.3236,0.12205],[52,-0.1366,17.5059,0.12320],[53,-0.1403,17.6893,0.12435],
    [54,-0.1440,17.8740,0.12548],[55,-0.1476,18.0599,0.12661],[56,-0.1512,18.2471,0.12773],
    [57,-0.1547,18.4355,0.12884],[58,-0.1582,18.6250,0.12993],[59,-0.1616,18.8157,0.13101],
    [60,-0.1650,19.0074,0.13208]
]

WFA_GIRLS = [
    [0,0.3809,3.2322,0.14171],[1,0.1714,4.1873,0.13724],[2,0.0962,5.1282,0.13000],
    [3,0.0569,5.8458,0.12516],[4,0.0372,6.4237,0.12059],[5,0.0247,6.8985,0.11750],
    [6,0.0150,7.2970,0.11556],[7,0.0062,7.6422,0.11501],[8,-0.0015,7.9350,0.11396],
    [9,-0.0093,8.2003,0.11329],[10,-0.0170,8.4505,0.11285],[11,-0.0244,8.6896,0.11296],
    [12,-0.0315,8.9481,0.11382],[13,-0.0385,9.2060,0.11473],[14,-0.0452,9.4484,0.11565],
    [15,-0.0517,9.6792,0.11678],[16,-0.0580,9.9037,0.11782],[17,-0.0642,10.1231,0.11899],
    [18,-0.0702,10.3409,0.12012],[19,-0.0760,10.5561,0.12107],[20,-0.0817,10.7706,0.12207],
    [21,-0.0873,10.9806,0.12316],[22,-0.0928,11.1894,0.12430],[23,-0.0981,11.3972,0.12545],
    [24,-0.1033,11.6031,0.12659],[25,-0.1084,11.8085,0.12776],[26,-0.1134,12.0124,0.12887],
    [27,-0.1183,12.2143,0.13001],[28,-0.1230,12.4148,0.13112],[29,-0.1277,12.6129,0.13218],
    [30,-0.1323,12.8089,0.13326],[31,-0.1368,13.0028,0.13431],[32,-0.1412,13.1940,0.13533],
    [33,-0.1455,13.3822,0.13634],[34,-0.1497,13.5680,0.13732],[35,-0.1538,13.7525,0.13828],
    [36,-0.1578,13.9348,0.13921],[37,-0.1618,14.1148,0.14012],[38,-0.1656,14.2939,0.14100],
    [39,-0.1694,14.4720,0.14188],[40,-0.1731,14.6498,0.14275],[41,-0.1767,14.8263,0.14359],
    [42,-0.1802,15.0021,0.14441],[43,-0.1836,15.1783,0.14523],[44,-0.1869,15.3542,0.14604],
    [45,-0.1902,15.5300,0.14682],[46,-0.1934,15.7058,0.14759],[47,-0.1965,15.8821,0.14836],
    [48,-0.1995,16.0586,0.14910],[49,-0.2025,16.2356,0.14983],[50,-0.2054,16.4133,0.15055],
    [51,-0.2082,16.5917,0.15127],[52,-0.2110,16.7706,0.15196],[53,-0.2137,16.9502,0.15264],
    [54,-0.2163,17.1305,0.15331],[55,-0.2189,17.3117,0.15397],[56,-0.2214,17.4938,0.15461],
    [57,-0.2238,17.6765,0.15524],[58,-0.2262,17.8601,0.15585],[59,-0.2285,18.0445,0.15646],
    [60,-0.2308,18.2298,0.15705]
]

WFL_BOYS = [
    [45.0,-0.3521,2.441,0.09182],[45.5,-0.3521,2.504,0.09076],[46.0,-0.3521,2.570,0.08969],
    [46.5,-0.3521,2.640,0.08864],[47.0,-0.3521,2.713,0.08763],[47.5,-0.3521,2.790,0.08664],
    [48.0,-0.3521,2.870,0.08570],[48.5,-0.3521,2.953,0.08477],[49.0,-0.3521,3.040,0.08388],
    [49.5,-0.3521,3.130,0.08303],[50.0,-0.3521,3.223,0.08221],[50.5,-0.3521,3.319,0.08143],
    [51.0,-0.3521,3.418,0.08071],[51.5,-0.3521,3.520,0.08003],[52.0,-0.3521,3.625,0.07939],
    [52.5,-0.3521,3.733,0.07881],[53.0,-0.3521,3.843,0.07827],[53.5,-0.3521,3.956,0.07777],
    [54.0,-0.3521,4.073,0.07732],[54.5,-0.3521,4.191,0.07691],[55.0,-0.3521,4.313,0.07654],
    [55.5,-0.3521,4.437,0.07621],[56.0,-0.3521,4.563,0.07593],[56.5,-0.3521,4.691,0.07569],
    [57.0,-0.3521,4.820,0.07548],[57.5,-0.3521,4.952,0.07531],[58.0,-0.3521,5.085,0.07517],
    [58.5,-0.3521,5.219,0.07507],[59.0,-0.3521,5.354,0.07500],[59.5,-0.3521,5.489,0.07496],
    [60.0,-0.3521,5.625,0.07495],[60.5,-0.3521,5.761,0.07498],[61.0,-0.3521,5.896,0.07503],
    [61.5,-0.3521,6.031,0.07512],[62.0,-0.3521,6.165,0.07524],[62.5,-0.3521,6.298,0.07538],
    [63.0,-0.3521,6.430,0.07556],[63.5,-0.3521,6.560,0.07575],[64.0,-0.3521,6.689,0.07598],
    [64.5,-0.3521,6.816,0.07624],[65.0,-0.3521,6.942,0.07651],[65.5,-0.3521,7.066,0.07682],
    [66.0,-0.3521,7.189,0.07715],[66.5,-0.3521,7.310,0.07750],[67.0,-0.3521,7.429,0.07787],
    [67.5,-0.3521,7.546,0.07827],[68.0,-0.3521,7.661,0.07869],[68.5,-0.3521,7.774,0.07914],
    [69.0,-0.3521,7.886,0.07960],[69.5,-0.3521,7.995,0.08008],[70.0,-0.3521,8.102,0.08058],
    [70.5,-0.3521,8.207,0.08109],[71.0,-0.3521,8.310,0.08162],[71.5,-0.3521,8.411,0.08217],
    [72.0,-0.3521,8.511,0.08274],[72.5,-0.3521,8.609,0.08332],[73.0,-0.3521,8.706,0.08392],
    [73.5,-0.3521,8.801,0.08454],[74.0,-0.3521,8.894,0.08517],[74.5,-0.3521,8.986,0.08582],
    [75.0,-0.3521,9.077,0.08648],[75.5,-0.3521,9.166,0.08716],[76.0,-0.3521,9.254,0.08784],
    [76.5,-0.3521,9.340,0.08855],[77.0,-0.3521,9.425,0.08926],[77.5,-0.3521,9.509,0.08998],
    [78.0,-0.3521,9.591,0.09072],[78.5,-0.3521,9.673,0.09147],[79.0,-0.3521,9.753,0.09222],
    [79.5,-0.3521,9.832,0.09299],[80.0,-0.3521,9.910,0.09376],[80.5,-0.3521,9.987,0.09454],
    [81.0,-0.3521,10.063,0.09532],[81.5,-0.3521,10.138,0.09610],[82.0,-0.3521,10.213,0.09689],
    [82.5,-0.3521,10.287,0.09768],[83.0,-0.3521,10.359,0.09847],[83.5,-0.3521,10.432,0.09926],
    [84.0,-0.3521,10.504,0.10005],[84.5,-0.3521,10.576,0.10084],[85.0,-0.3521,10.647,0.10163],
    [85.5,-0.3521,10.718,0.10242],[86.0,-0.3521,10.789,0.10320],[86.5,-0.3521,10.859,0.10398],
    [87.0,-0.3521,10.929,0.10476],[87.5,-0.3521,10.999,0.10553],[88.0,-0.3521,11.069,0.10630],
    [88.5,-0.3521,11.139,0.10707],[89.0,-0.3521,11.209,0.10783],[89.5,-0.3521,11.279,0.10858],
    [90.0,-0.3521,11.349,0.10933],[90.5,-0.3521,11.420,0.11007],[91.0,-0.3521,11.490,0.11081],
    [91.5,-0.3521,11.561,0.11154],[92.0,-0.3521,11.632,0.11226],[92.5,-0.3521,11.703,0.11298],
    [93.0,-0.3521,11.774,0.11369],[93.5,-0.3521,11.845,0.11440],[94.0,-0.3521,11.917,0.11510],
    [94.5,-0.3521,11.989,0.11580],[95.0,-0.3521,12.062,0.11649],[95.5,-0.3521,12.135,0.11718],
    [96.0,-0.3521,12.208,0.11786],[96.5,-0.3521,12.281,0.11854],[97.0,-0.3521,12.355,0.11922],
    [97.5,-0.3521,12.430,0.11990],[98.0,-0.3521,12.506,0.12057],[98.5,-0.3521,12.582,0.12124],
    [99.0,-0.3521,12.659,0.12191],[99.5,-0.3521,12.737,0.12258],[100.0,-0.3521,12.816,0.12325],
    [100.5,-0.3521,12.896,0.12393],[101.0,-0.3521,12.976,0.12460],[101.5,-0.3521,13.057,0.12528],
    [102.0,-0.3521,13.139,0.12596],[102.5,-0.3521,13.222,0.12664],[103.0,-0.3521,13.305,0.12733],
    [103.5,-0.3521,13.389,0.12802],[104.0,-0.3521,13.474,0.12872],[104.5,-0.3521,13.560,0.12942],
    [105.0,-0.3521,13.646,0.13012],[105.5,-0.3521,13.733,0.13083],[106.0,-0.3521,13.821,0.13154],
    [106.5,-0.3521,13.909,0.13226],[107.0,-0.3521,13.998,0.13298],[107.5,-0.3521,14.088,0.13370],
    [108.0,-0.3521,14.178,0.13444],[108.5,-0.3521,14.269,0.13517],[109.0,-0.3521,14.361,0.13591],
    [109.5,-0.3521,14.453,0.13666],[110.0,-0.3521,14.546,0.13741]
]

WFL_GIRLS = [
    [45.0,-0.3833,2.460,0.09029],[45.5,-0.3833,2.524,0.08928],[46.0,-0.3833,2.591,0.08825],
    [46.5,-0.3833,2.660,0.08720],[47.0,-0.3833,2.732,0.08614],[47.5,-0.3833,2.807,0.08509],
    [48.0,-0.3833,2.884,0.08405],[48.5,-0.3833,2.964,0.08302],[49.0,-0.3833,3.047,0.08201],
    [49.5,-0.3833,3.132,0.08103],[50.0,-0.3833,3.220,0.08007],[50.5,-0.3833,3.311,0.07914],
    [51.0,-0.3833,3.405,0.07824],[51.5,-0.3833,3.502,0.07737],[52.0,-0.3833,3.601,0.07653],
    [52.5,-0.3833,3.703,0.07573],[53.0,-0.3833,3.808,0.07497],[53.5,-0.3833,3.914,0.07424],
    [54.0,-0.3833,4.023,0.07355],[54.5,-0.3833,4.134,0.07289],[55.0,-0.3833,4.247,0.07228],
    [55.5,-0.3833,4.360,0.07170],[56.0,-0.3833,4.474,0.07116],[56.5,-0.3833,4.589,0.07065],
    [57.0,-0.3833,4.705,0.07019],[57.5,-0.3833,4.822,0.06976],[58.0,-0.3833,4.939,0.06938],
    [58.5,-0.3833,5.057,0.06903],[59.0,-0.3833,5.175,0.06871],[59.5,-0.3833,5.293,0.06843],
    [60.0,-0.3833,5.411,0.06818],[60.5,-0.3833,5.528,0.06796],[61.0,-0.3833,5.644,0.06777],
    [61.5,-0.3833,5.760,0.06761],[62.0,-0.3833,5.875,0.06748],[62.5,-0.3833,5.989,0.06737],
    [63.0,-0.3833,6.102,0.06729],[63.5,-0.3833,6.213,0.06724],[64.0,-0.3833,6.323,0.06720],
    [64.5,-0.3833,6.432,0.06719],[65.0,-0.3833,6.540,0.06720],[65.5,-0.3833,6.646,0.06724],
    [66.0,-0.3833,6.751,0.06730],[66.5,-0.3833,6.855,0.06737],[67.0,-0.3833,6.958,0.06746],
    [67.5,-0.3833,7.059,0.06757],[68.0,-0.3833,7.159,0.06771],[68.5,-0.3833,7.257,0.06786],
    [69.0,-0.3833,7.354,0.06802],[69.5,-0.3833,7.449,0.06820],[70.0,-0.3833,7.543,0.06840],
    [70.5,-0.3833,7.635,0.06861],[71.0,-0.3833,7.726,0.06884],[71.5,-0.3833,7.816,0.06908],
    [72.0,-0.3833,7.904,0.06934],[72.5,-0.3833,7.991,0.06961],[73.0,-0.3833,8.076,0.06990],
    [73.5,-0.3833,8.161,0.07020],[74.0,-0.3833,8.244,0.07052],[74.5,-0.3833,8.326,0.07085],
    [75.0,-0.3833,8.407,0.07120],[75.5,-0.3833,8.488,0.07156],[76.0,-0.3833,8.567,0.07194],
    [76.5,-0.3833,8.645,0.07233],[77.0,-0.3833,8.723,0.07274],[77.5,-0.3833,8.800,0.07316],
    [78.0,-0.3833,8.876,0.07360],[78.5,-0.3833,8.951,0.07405],[79.0,-0.3833,9.026,0.07451],
    [79.5,-0.3833,9.100,0.07499],[80.0,-0.3833,9.174,0.07548],[80.5,-0.3833,9.247,0.07598],
    [81.0,-0.3833,9.319,0.07650],[81.5,-0.3833,9.391,0.07703],[82.0,-0.3833,9.463,0.07758],
    [82.5,-0.3833,9.535,0.07814],[83.0,-0.3833,9.607,0.07870],[83.5,-0.3833,9.678,0.07928],
    [84.0,-0.3833,9.750,0.07988],[84.5,-0.3833,9.822,0.08049],[85.0,-0.3833,9.894,0.08111],
    [85.5,-0.3833,9.966,0.08174],[86.0,-0.3833,10.038,0.08238],[86.5,-0.3833,10.111,0.08304],
    [87.0,-0.3833,10.184,0.08371],[87.5,-0.3833,10.258,0.08439],[88.0,-0.3833,10.333,0.08508],
    [88.5,-0.3833,10.408,0.08579],[89.0,-0.3833,10.483,0.08651],[89.5,-0.3833,10.559,0.08724],
    [90.0,-0.3833,10.635,0.08798],[90.5,-0.3833,10.712,0.08874],[91.0,-0.3833,10.789,0.08951],
    [91.5,-0.3833,10.867,0.09029],[92.0,-0.3833,10.946,0.09108],[92.5,-0.3833,11.025,0.09188],
    [93.0,-0.3833,11.104,0.09269],[93.5,-0.3833,11.184,0.09351],[94.0,-0.3833,11.264,0.09434],
    [94.5,-0.3833,11.344,0.09517],[95.0,-0.3833,11.425,0.09601],[95.5,-0.3833,11.505,0.09686],
    [96.0,-0.3833,11.586,0.09772],[96.5,-0.3833,11.668,0.09858],[97.0,-0.3833,11.749,0.09945],
    [97.5,-0.3833,11.832,0.10033],[98.0,-0.3833,11.915,0.10122],[98.5,-0.3833,11.998,0.10212],
    [99.0,-0.3833,12.082,0.10302],[99.5,-0.3833,12.167,0.10393],[100.0,-0.3833,12.252,0.10485],
    [100.5,-0.3833,12.338,0.10578],[101.0,-0.3833,12.424,0.10671],[101.5,-0.3833,12.511,0.10765],
    [102.0,-0.3833,12.599,0.10860],[102.5,-0.3833,12.688,0.10956],[103.0,-0.3833,12.777,0.11053],
    [103.5,-0.3833,12.866,0.11150],[104.0,-0.3833,12.957,0.11248],[104.5,-0.3833,13.048,0.11347],
    [105.0,-0.3833,13.140,0.11447],[105.5,-0.3833,13.232,0.11548],[106.0,-0.3833,13.325,0.11650],
    [106.5,-0.3833,13.419,0.11752],[107.0,-0.3833,13.513,0.11856],[107.5,-0.3833,13.608,0.11960],
    [108.0,-0.3833,13.704,0.12065],[108.5,-0.3833,13.800,0.12170],[109.0,-0.3833,13.897,0.12276],
    [109.5,-0.3833,13.995,0.12383],[110.0,-0.3833,14.093,0.12490]
]


# =====================================================================
# WHO Z-SCORE FUNCTIONS
# =====================================================================

def lms_zscore(X, L, M, S):
    if L == 0:
        z = np.log(X / M) / S
    else:
        z = ((X / M) ** L - 1) / (L * S)
    if z > 3:
        SD3pos  = M * (1 + L * S * 3) ** (1 / L)
        SD23pos = SD3pos - M * (1 + L * S * 2) ** (1 / L)
        z = 3 + (X - SD3pos) / SD23pos
    elif z < -3:
        SD3neg  = M * (1 + L * S * (-3)) ** (1 / L)
        SD23neg = M * (1 + L * S * (-2)) ** (1 / L) - SD3neg
        z = -3 + (X - SD3neg) / SD23neg
    return round(z, 2)


def get_lms_by_age(age_months, sex):
    table = WFA_BOYS if sex == 'L' else WFA_GIRLS
    age = max(0, min(60, int(round(age_months))))
    for row in table:
        if int(row[0]) == age:
            return row[1], row[2], row[3]
    return table[-1][1], table[-1][2], table[-1][3]


def get_lms_by_height(height_cm, sex):
    table = WFL_BOYS if sex == 'L' else WFL_GIRLS
    heights = [r[0] for r in table]
    h = max(heights[0], min(heights[-1], height_cm))
    idx = bisect.bisect_left(heights, h)
    if idx >= len(table):
        idx = len(table) - 1
    elif idx > 0:
        h0, h1 = heights[idx-1], heights[idx]
        t = (h - h0) / (h1 - h0) if h1 != h0 else 0
        L = table[idx-1][1] + t*(table[idx][1]-table[idx-1][1])
        M = table[idx-1][2] + t*(table[idx][2]-table[idx-1][2])
        S = table[idx-1][3] + t*(table[idx][3]-table[idx-1][3])
        return L, M, S
    return table[idx][1], table[idx][2], table[idx][3]


def compute_zscore_bbu(weight, age_months, sex):
    L, M, S = get_lms_by_age(age_months, sex)
    return lms_zscore(weight, L, M, S)


def compute_zscore_bbtb(weight, height_cm, sex, cara_ukur, age_months):
    h = height_cm
    if cara_ukur == 'Terlentang' and age_months >= 24:
        h = height_cm - 0.7
    elif cara_ukur == 'Berdiri' and age_months < 24:
        h = height_cm + 0.7
    L, M, S = get_lms_by_height(h, sex)
    return lms_zscore(weight, L, M, S)


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def build_features(umur_bulan, jk, berat, tinggi, cara_ukur):
    zs_bb_u  = compute_zscore_bbu(berat, umur_bulan, jk)
    zs_bb_tb = compute_zscore_bbtb(berat, tinggi, jk, cara_ukur, umur_bulan)

    jk_enc   = 1 if jk == 'L' else 0
    cara_enc = 1 if cara_ukur == 'Berdiri' else 0

    bins = [-1, 6, 11, 23, 36, 60]
    kel  = 0
    for i in range(len(bins)-1):
        if umur_bulan > bins[i] and umur_bulan <= bins[i+1]:
            kel = [0,1,2,3,4][i]

    f_window = 1 if umur_bulan <= 23 else 0
    f_mpasi  = 1 if 6 <= umur_bulan <= 23 else 0
    f_baduta = 1 if umur_bulan <= 23 else 0
    age_sq   = umur_bulan ** 2
    age_log  = np.log1p(umur_bulan)

    f_bb_sk = 1 if zs_bb_u < -3 else 0
    f_bb_k  = 1 if -3 <= zs_bb_u < -2 else 0
    f_bb_n  = 1 if -2 <= zs_bb_u <= 1 else 0
    f_bb_rl = 1 if zs_bb_u > 1 else 0
    f_uw    = 1 if zs_bb_u < -2 else 0

    f_gb  = 1 if zs_bb_tb < -3 else 0
    f_gk  = 1 if -3 <= zs_bb_tb < -2 else 0
    f_gbk = 1 if -2 <= zs_bb_tb <= 1 else 0
    f_rgl = 1 if 1 < zs_bb_tb <= 2 else 0
    f_gl  = 1 if 2 < zs_bb_tb <= 3 else 0
    f_ob  = 1 if zs_bb_tb > 3 else 0
    f_wst = 1 if zs_bb_tb < -2 else 0

    f_bb_mild   = 1 if -2.5 <= zs_bb_u < -2 else 0
    f_bbtb_mild = 1 if -2.5 <= zs_bb_tb < -2 else 0
    prox_bbu2   = max(0, zs_bb_u - (-2))
    prox_bbtb2  = max(0, zs_bb_tb - (-2))
    prox_bbu3   = max(0, zs_bb_u - (-3))
    prox_bbtb3  = max(0, zs_bb_tb - (-3))

    f_double = 1 if f_uw == 1 and f_wst == 1 else 0
    f_dsev   = 1 if f_bb_sk == 1 and f_gb == 1 else 0
    f_any_sv = 1 if f_bb_sk == 1 or f_gb == 1 else 0
    skor     = f_bb_sk*2 + f_bb_k + f_gb*2 + f_gk
    jml_idx  = (1 if f_uw else 0) + (1 if f_wst else 0)

    avg_z    = (zs_bb_u + zs_bb_tb) / 2
    min_z    = min(zs_bb_u, zs_bb_tb)
    max_z    = max(zs_bb_u, zs_bb_tb)
    gap      = zs_bb_u - zs_bb_tb
    abs_g    = abs(gap)
    prod     = zs_bb_u * zs_bb_tb
    harm     = (2*zs_bb_u*zs_bb_tb/(zs_bb_u+zs_bb_tb)) if (zs_bb_u!=0 and zs_bb_tb!=0) else 0.0
    bbu_sq   = zs_bb_u ** 2
    bbtb_sq  = zs_bb_tb ** 2
    bbu_cb   = zs_bb_u ** 3
    bbtb_cb  = zs_bb_tb ** 3

    umur_x_risiko = umur_bulan * skor
    umur_x_minzs  = umur_bulan * min_z
    umur_x_bbu    = umur_bulan * zs_bb_u
    umur_x_bbtb   = umur_bulan * zs_bb_tb
    bad_x_wst     = f_baduta * f_wst
    mpa_x_uw      = f_mpasi  * f_uw

    jk_x_minzs  = jk_enc * min_z
    jk_x_bbu    = jk_enc * zs_bb_u
    jk_x_risiko = jk_enc * skor

    bmi_proxy   = berat / (tinggi/100)**2
    rasio_bb_tb = berat / tinggi
    rasio_tb_um = tinggi / umur_bulan if umur_bulan > 0 else 0
    rasio_bb_um = berat  / umur_bulan if umur_bulan > 0 else 0

    rank_bbu  = float(1 / (1 + np.exp(-zs_bb_u)))
    rank_bbtb = float(1 / (1 + np.exp(-zs_bb_tb)))

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
        'umur_bulan','jk_encoded','cara_ukur_encoded',
        'kel_usia_permenkes','f_window_1000hpk','f_masa_mpasi','f_baduta',
        'age_sq','age_log','zs_bb_u',
        'f_bb_sangat_kurang','f_bb_kurang','f_bb_normal','f_bb_risiko_lebih','f_underweight',
        'zs_bb_tb','f_gizi_buruk','f_gizi_kurang','f_gizi_baik',
        'f_risiko_gizi_lebih','f_gizi_lebih','f_obesitas','f_wasting',
        'f_bb_mild','f_bbtb_mild','prox_bbu_ke_minus2','prox_bbtb_ke_minus2',
        'prox_bbu_ke_minus3','prox_bbtb_ke_minus3',
        'f_double_malnutrisi','f_double_severe','f_any_severe',
        'skor_risiko_gizi','jml_indeks_masalah',
        'avg_zs_bbu_bbtb','min_zs_bbu_bbtb','max_zs_bbu_bbtb',
        'gap_bbu_bbtb','abs_gap_bbu_bbtb','product_zs','harmonic_zs',
        'zs_bbu_sq','zs_bbtb_sq','zs_bbu_cb','zs_bbtb_cb',
        'umur_x_risiko','umur_x_minzs','umur_x_bbu','umur_x_bbtb',
        'baduta_x_wasting','mpasi_x_underw',
        'jk_x_minzs','jk_x_bbu','jk_x_risiko',
        'bmi_proxy','rasio_bb_tb','rasio_tb_umur','rasio_bb_umur',
        'berat','tinggi','rank_bbu','rank_bbtb'
    ]
    df = pd.DataFrame([row])[features]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, zs_bb_u, zs_bb_tb


# =====================================================================
# MODEL LOADER
# =====================================================================

class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights if weights else [1/len(models)] * len(models)

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
    # model.pkl adalah dict {model, threshold, features, weights}
    if isinstance(data, dict):
        model     = data["model"]
        threshold = float(data.get("threshold", 0.45))
        features  = data.get("features", None)
    else:
        model     = data
        threshold = 0.45
        features  = None
    if features is None:
        try:
            features = joblib.load("fitur_training.pkl")
        except Exception:
            features = []
    return model, threshold, features


# =====================================================================
# ── HERO SECTION ──
# =====================================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-orb"></div>
    <div class="hero-badge">🌿 &nbsp; Sistem Skrining AI · Berbasis Machine Learning</div>
    <h1 class="hero-title">Prediksi <em>Stunting</em> Balita</h1>
    <div class="hero-divider"></div>
    <p class="hero-sub">
        Analisis antropometri menggunakan WHO 2006 Growth Standards<br>
        Sesuai Permenkes No. 2 Tahun 2020 &nbsp;·&nbsp; Ensemble CatBoost + XGBoost
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# ── INPUT FORM ──
# =====================================================================
st.markdown('<p class="section-label">Data Antropometri Anak</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    umur_bulan = st.number_input(
        "Umur (Bulan)", min_value=0, max_value=60, value=24, step=1,
        help="Usia anak 0–60 bulan"
    )
    berat = st.number_input(
        "Berat Badan (kg)", min_value=1.0, max_value=35.0,
        value=12.0, step=0.1, format="%.1f"
    )
    jenis_kelamin = st.selectbox(
        "Jenis Kelamin", options=["Laki-laki", "Perempuan"]
    )
with col2:
    tinggi = st.number_input(
        "Tinggi / Panjang Badan (cm)", min_value=40.0, max_value=130.0,
        value=87.0, step=0.1, format="%.1f"
    )
    cara_ukur = st.selectbox(
        "Cara Pengukuran",
        options=["Terlentang — Panjang Badan", "Berdiri — Tinggi Badan"],
        help="Terlentang < 24 bln  ·  Berdiri ≥ 24 bln"
    )

# Guidance warning
cu_mode = "Berdiri" if cara_ukur.startswith("Berdiri") else "Terlentang"
if umur_bulan < 24 and cu_mode == "Berdiri":
    st.warning("Untuk usia < 24 bulan, disarankan pengukuran Terlentang (panjang badan).")
elif umur_bulan >= 24 and cu_mode == "Terlentang":
    st.warning("Untuk usia ≥ 24 bulan, disarankan pengukuran Berdiri (tinggi badan).")

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
predict_btn = st.button("Analisis & Prediksi Sekarang")


# =====================================================================
# ── RESULTS ──
# =====================================================================
if predict_btn:
    jk = "L" if jenis_kelamin == "Laki-laki" else "P"

    with st.spinner("Menganalisis data antropometri..."):
        try:
            model, threshold, _ = load_model()
            X_input, zs_bbu, zs_bbtb = build_features(
                umur_bulan, jk, berat, tinggi, cu_mode
            )
            proba         = model.predict_proba(X_input)[0]
            prob_stunting = proba[1]
            prediction    = 1 if prob_stunting >= threshold else 0
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")
            st.stop()

    pct = prob_stunting * 100
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Hasil Analisis</p>', unsafe_allow_html=True)

    # ── RESULT PANEL ──
    # ── 4-LEVEL RISK TIERING BERDASARKAN PROBABILITAS ──
    # Tampilan ditentukan murni dari nilai probabilitas (bukan binary threshold model)
    # sehingga hasil proporsional dan intuitif bagi pengguna.
    # Batas wajar: < 45% = aman, 45-65% = waspada, 65-80% = sedang, >80% = tinggi
    if pct < 45:
        # Probabilitas rendah → tumbuh kembang aman
        rp_cls   = "rp-normal"
        rp_icon  = "✓"
        rp_title = "Tumbuh Kembang Normal"
        rp_color = "#34d399"
        rp_eyeb  = "Status Pertumbuhan · Aman"
        rec_main = "Tumbuh kembang anak berada dalam batas normal. Pertahankan asupan gizi seimbang, ASI/MPASI sesuai usia, dan lakukan pemantauan rutin di posyandu."
    elif pct < 65:
        # Probabilitas sedang-bawah → perlu diwaspadai
        rp_cls   = "rp-waspada"
        rp_icon  = "○"
        rp_title = "Perlu Diwaspadai"
        rp_color = "#facc15"
        rp_eyeb  = "Status Pertumbuhan · Perlu Perhatian"
        rec_main = "Terdapat indikasi awal risiko. Perhatikan kecukupan asupan gizi dan jadwal pemantauan di posyandu. Konsultasikan ke petugas gizi atau bidan untuk evaluasi lebih lanjut."
    elif pct < 80:
        # Probabilitas sedang-atas → risiko sedang
        rp_cls   = "rp-sedang"
        rp_icon  = "⚡"
        rp_title = "Risiko Stunting Sedang"
        rp_color = "#fb923c"
        rp_eyeb  = "Perhatian · Intervensi Gizi Dianjurkan"
        rec_main = "Risiko stunting terdeteksi pada level sedang. Segera konsultasikan ke dokter atau ahli gizi untuk evaluasi pola makan, asupan protein, dan stimulasi tumbuh kembang."
    else:
        # Probabilitas tinggi → risiko tinggi
        rp_cls   = "rp-stunting"
        rp_icon  = "⚠️"
        rp_title = "Risiko Stunting Tinggi"
        rp_color = "#f87171"
        rp_eyeb  = "Perhatian · Tindak Lanjut Segera Diperlukan"
        rec_main = "Risiko stunting terdeteksi pada level tinggi. Segera bawa ke fasilitas kesehatan (puskesmas/dokter spesialis anak) untuk intervensi gizi intensif."

    st.markdown(f"""
    <div class="result-panel {rp_cls}">
        <div class="rp-eyebrow" style="color:{rp_color};">{rp_eyeb}</div>
        <div class="rp-icon">{rp_icon}</div>
        <div class="rp-headline" style="color:{rp_color};">{rp_title}</div>
        <div class="rp-prob">Probabilitas stunting &nbsp;&mdash;&nbsp;
            <strong style="color:{rp_color}; font-size:1rem;">{pct:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Z-SCORE CARDS ──
    def bbu_style(z):
        if z < -3:  return "Sangat Kurang", "#f87171", "rgba(248,113,113,.14)"
        if z < -2:  return "Kurang",         "#fb923c", "rgba(251,146,60,.14)"
        if z <= 1:  return "Normal",          "#34d399", "rgba(52,211,153,.14)"
        return             "Risiko Lebih",   "#facc15", "rgba(250,204,21,.14)"

    def bbtb_style(z):
        if z < -3:  return "Gizi Buruk",     "#f87171", "rgba(248,113,113,.14)"
        if z < -2:  return "Gizi Kurang",    "#fb923c", "rgba(251,146,60,.14)"
        if z <= 1:  return "Gizi Baik",      "#34d399", "rgba(52,211,153,.14)"
        if z <= 2:  return "Risiko Lebih",   "#facc15", "rgba(250,204,21,.14)"
        if z <= 3:  return "Gizi Lebih",     "#f97316", "rgba(249,115,22,.14)"
        return             "Obesitas",       "#ef4444", "rgba(239,68,68,.14)"

    bc, bcol, bbg    = bbu_style(zs_bbu)
    btbc, btbcol, btbbg = bbtb_style(zs_bbtb)

    st.markdown(f"""
    <div class="zs-grid">
        <div class="zs-card">
            <div class="zs-tag">Z-Score BB / Umur</div>
            <div class="zs-val" style="color:{bcol};">{zs_bbu:+.2f}</div>
            <div class="zs-desc">Berat Badan menurut Umur</div>
            <span class="zs-badge" style="background:{bbg}; color:{bcol};">{bc}</span>
        </div>
        <div class="zs-card">
            <div class="zs-tag">Z-Score BB / Tinggi</div>
            <div class="zs-val" style="color:{btbcol};">{zs_bbtb:+.2f}</div>
            <div class="zs-desc">Berat Badan menurut Tinggi</div>
            <span class="zs-badge" style="background:{btbbg}; color:{btbcol};">{btbc}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PROBABILITY GAUGE ──
    fill_col = rp_color
    bar_pct  = max(2, min(97, pct))

    st.markdown(f"""
    <div class="gauge-wrap">
        <div class="gauge-top">
            <span class="gauge-ttl">Indeks Probabilitas Stunting</span>
            <span class="gauge-num" style="color:{fill_col};">
                {pct:.1f}<span style="font-size:.9rem; opacity:.55;">%</span>
            </span>
        </div>
        <div class="gauge-track">
            <div class="gauge-bar" style="width:100%;"></div>
            <div class="gauge-pip" style="left:{bar_pct}%; color:{fill_col}; background:{fill_col};"></div>
        </div>
        <div class="gauge-ticks">
            <span>Normal</span>
            <span>Ambang batas 45%</span>
            <span>Stunting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── META PILLS ──
    bmi_val = berat / (tinggi/100)**2
    kel_map = {0:"ASI Eksklusif", 1:"MPASI Awal", 2:"Baduta / 1000 HPK", 3:"Batita", 4:"Balita"}
    bins_k  = [-1,6,11,23,36,60]
    kel     = 0
    for i in range(len(bins_k)-1):
        if umur_bulan > bins_k[i] and umur_bulan <= bins_k[i+1]:
            kel = [0,1,2,3,4][i]

    is_1000hpk = umur_bulan <= 23
    hpk_col  = "#34d399" if is_1000hpk else "#64748b"
    hpk_text = "Aktif" if is_1000hpk else "Telah Lewat"

    st.markdown(f"""
    <div class="meta-row">
        <div class="meta-pill">
            <div class="mp-label">BMI Anak</div>
            <div class="mp-val">{bmi_val:.1f}</div>
            <div class="mp-sub">kg / m²</div>
        </div>
        <div class="meta-pill">
            <div class="mp-label">Kelompok Usia</div>
            <div class="mp-val" style="font-size:1rem; padding:.3rem 0;">{kel_map.get(kel,'—')}</div>
            <div class="mp-sub">Permenkes No.2/2020</div>
        </div>
        <div class="meta-pill">
            <div class="mp-label">Window 1000 HPK</div>
            <div class="mp-val" style="color:{hpk_col}; font-size:1.1rem;">{hpk_text}</div>
            <div class="mp-sub">0 – 23 bulan kritis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── RECOMMENDATION ──
    st.markdown(f"""
    <div class="rec-strip">
        <span class="rec-acc">Rekomendasi — </span>{rec_main}
        <div class="rec-disc">
            Hasil ini merupakan skrining awal berbasis AI dan tidak menggantikan diagnosis medis profesional.
            Selalu konsultasikan ke dokter, bidan, atau tenaga kesehatan untuk penilaian klinis yang menyeluruh.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# ── FOOTER ──
# =====================================================================
st.markdown("""
<div class="footer-note">
    <span>Permenkes No. 2 Tahun 2020</span> — Standar Antropometri Anak
    <span class="footer-dot">·</span>
    <span>WHO 2006 Multicentre Growth Reference Study</span>
    <br>Model Ensemble CatBoost + XGBoost
    <span class="footer-dot">·</span>
    Skrining Awal — Bukan Pengganti Diagnosis Medis
</div>
""", unsafe_allow_html=True)
