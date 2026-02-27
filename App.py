import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bisect
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Prediksi Stunting Balita",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --navy:  #0b1120;
    --card:  #111827;
    --card2: #1a2535;
    --gold:  #c9a84c;
    --gold-lt: #e8d48b;
    --gold-dim: rgba(201,168,76,.15);
    --t1: #f1f5f9; --t2: #94a3b8; --t3: #64748b;
    --border: rgba(255,255,255,.08);
    --border-gold: rgba(201,168,76,.3);
}

html, body, [class*="css"], .stApp, .main,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--t1) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 820px !important; }

/* ── SECTION LABEL ── */
.slabel {
    font-size: .62rem; font-weight: 600; letter-spacing: .22em;
    text-transform: uppercase; color: var(--gold);
    display: flex; align-items: center; gap: .7rem;
    margin: 1.4rem 0 1rem;
}
.slabel::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-gold), transparent);
}

/* ── INPUTS ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1.5px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: var(--t1) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: .95rem !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-dim) !important;
}
div[data-testid="stSelectbox"] ul {
    background: var(--card2) !important;
    border: 1px solid var(--border-gold) !important;
    border-radius: 10px !important;
}
div[data-testid="stSelectbox"] li {
    border-radius: 7px !important; color: var(--t2) !important;
}
div[data-testid="stSelectbox"] li:hover {
    background: var(--gold-dim) !important; color: var(--gold-lt) !important;
}
label[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] p {
    color: var(--t3) !important; font-size: .75rem !important;
    font-weight: 600 !important; letter-spacing: .15em !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] button {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--t3) !important; border-radius: 7px !important;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #9a6e1a, var(--gold) 50%, var(--gold-lt)) !important;
    color: #0b1120 !important; border: none !important;
    border-radius: 12px !important; padding: .9rem 2rem !important;
    font-size: .88rem !important; font-weight: 600 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important;
    font-family: 'Outfit', sans-serif !important;
    box-shadow: 0 6px 22px rgba(201,168,76,.28) !important;
    transition: all .25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(201,168,76,.38) !important;
}

/* ── RESULT CARD ── */
.rcard {
    border-radius: 18px; padding: 2rem 1.5rem;
    text-align: center; position: relative;
    overflow: hidden; margin-bottom: 1.2rem;
    animation: pop .5s cubic-bezier(.34,1.4,.64,1) both;
}
.rcard::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 65% 45% at 50% 0%, var(--glow) 0%, transparent 70%);
}
.rc-normal  { background:linear-gradient(160deg,#091a12,#0d2218); border:1px solid rgba(52,211,153,.25); --glow:rgba(52,211,153,.12); }
.rc-waspada { background:linear-gradient(160deg,#1a1400,#221a00); border:1px solid rgba(250,204,21,.25); --glow:rgba(250,204,21,.10); }
.rc-sedang  { background:linear-gradient(160deg,#1a0e00,#221400); border:1px solid rgba(251,146,60,.25); --glow:rgba(251,146,60,.11); }
.rc-tinggi  { background:linear-gradient(160deg,#1a0909,#220d0d); border:1px solid rgba(248,113,113,.25); --glow:rgba(248,113,113,.12); }
.rc-eyebrow { font-size:.62rem; font-weight:600; letter-spacing:.2em; text-transform:uppercase; opacity:.65; margin-bottom:.7rem; position:relative; }
.rc-check   { font-size:2rem; margin-bottom:.2rem; position:relative; line-height:1; }
.rc-title   { font-family:'Cormorant Garamond',serif; font-size:2.3rem; font-weight:400; letter-spacing:-.02em; line-height:1; margin-bottom:.4rem; position:relative; }
.rc-prob    { font-size:.83rem; opacity:.6; position:relative; }

/* ── Z-SCORE CARDS ── */
.zrow { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.8rem; margin-bottom:1rem; }
.zcard {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.2rem .9rem; text-align:center;
    transition:border-color .2s, transform .15s;
}
.zcard:hover { border-color:var(--border-gold); transform:translateY(-2px); }
.ztag  { font-size:.6rem; letter-spacing:.13em; text-transform:uppercase; font-weight:600; color:var(--t3); margin-bottom:.5rem; }
.zval  { font-family:'Cormorant Garamond',serif; font-size:2.5rem; font-weight:300; line-height:1; margin-bottom:.35rem; }
.zdesc { font-size:.68rem; color:var(--t3); margin-bottom:.4rem; }
.zbadge { font-size:.7rem; font-weight:500; padding:.22rem .65rem; border-radius:100px; display:inline-block; }

/* ── GAUGE ── */
.gauge {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:1rem;
}
.gtop  { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.8rem; }
.gttl  { font-size:.62rem; letter-spacing:.15em; text-transform:uppercase; font-weight:600; color:var(--t3); }
.gnum  { font-family:'Cormorant Garamond',serif; font-size:1.5rem; font-weight:300; }
.gtrack { height:6px; border-radius:100px; background:rgba(255,255,255,.07); position:relative; }
.gfill  { position:absolute; left:0; top:0; height:100%; border-radius:100px; background:linear-gradient(90deg,#34d399 0%,#facc15 45%,#fb923c 70%,#f87171 100%); width:100%; }
.gpip   { position:absolute; top:50%; transform:translate(-50%,-50%); width:12px; height:12px; border-radius:50%; border:2px solid var(--navy); z-index:2; box-shadow:0 0 8px currentColor; }
.gtick  { display:flex; justify-content:space-between; margin-top:.45rem; font-size:.64rem; color:var(--t3); }

/* ── RECOMMENDATION ── */
.recbox {
    background:var(--card); border:1px solid var(--border);
    border-left:3px solid; border-radius:14px;
    padding:1rem 1.2rem; font-size:.86rem; line-height:1.75;
    color:var(--t2); margin-bottom:1rem;
}
.rec-item { display:flex; align-items:flex-start; gap:.45rem; margin:.2rem 0; }
.rec-item::before { content:'→'; color:var(--gold); flex-shrink:0; }
.rec-disc { font-size:.73rem; color:var(--t3); margin-top:.55rem; font-style:italic; }

/* ── ALERTS ── */
div[data-testid="stAlert"] {
    background:rgba(201,168,76,.07) !important;
    border:1px solid var(--border-gold) !important;
    border-radius:10px !important; color:var(--gold-lt) !important;
}

/* ── ANIMATIONS ── */
@keyframes pop { from{opacity:0;transform:scale(.94);} to{opacity:1;transform:scale(1);} }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WHO LMS — BB/U
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
# WHO LMS — BB/TB
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
# WHO SD — TB/U  [age, -3SD, -2SD, median, +2SD, +3SD]
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
# Z-SCORE FUNCTIONS
# ─────────────────────────────────────────────
def lms_zscore(X, L, M, S):
    if L == 0:
        z = np.log(X / M) / S
    else:
        z = ((X / M) ** L - 1) / (L * S)
    if z > 3:
        SD3  = M * (1 + L*S*3)**(1/L)
        SD23 = SD3 - M*(1+L*S*2)**(1/L)
        z = 3 + (X - SD3)/SD23 if SD23 else 3
    elif z < -3:
        SD3  = M * (1+L*S*(-3))**(1/L)
        SD23 = M*(1+L*S*(-2))**(1/L) - SD3
        z = -3 + (X - SD3)/SD23 if SD23 else -3
    return round(z, 2)

def interp_lms_age(age, table):
    ages = [r[0] for r in table]
    a = max(0, min(60, int(round(age))))
    idx = bisect.bisect_left(ages, a)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0 and ages[idx] != a:
        a0, a1 = ages[idx-1], ages[idx]
        t = (a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,4))
    return table[idx][1], table[idx][2], table[idx][3]

def interp_lms_height(h, table):
    hs = [r[0] for r in table]
    h  = max(hs[0], min(hs[-1], h))
    idx = bisect.bisect_left(hs, h)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0:
        h0, h1 = hs[idx-1], hs[idx]
        t = (h-h0)/(h1-h0) if h1!=h0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,4))
    return table[idx][1], table[idx][2], table[idx][3]

def interp_hfa(age, table):
    ages = [r[0] for r in table]
    a = max(0, min(60, age))
    idx = bisect.bisect_left(ages, a)
    if idx >= len(table): idx = len(table)-1
    elif idx > 0 and ages[idx] != a:
        a0, a1 = ages[idx-1], ages[idx]
        t = (a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(table[idx-1][i]+t*(table[idx][i]-table[idx-1][i]) for i in range(1,6))
    return tuple(table[idx][1:6])

def calc_bbu(weight, age, sex):
    L, M, S = interp_lms_age(age, WFA_BOYS if sex=='L' else WFA_GIRLS)
    return lms_zscore(weight, L, M, S)

def calc_bbtb(weight, h, sex, cara, age):
    hh = h
    if cara == 'Terlentang' and age >= 24: hh = h - 0.7
    elif cara == 'Berdiri'  and age <  24: hh = h + 0.7
    L, M, S = interp_lms_height(hh, WFL_BOYS if sex=='L' else WFL_GIRLS)
    return lms_zscore(weight, L, M, S)

def calc_tbu(height, age, sex):
    s3n, s2n, med, s2p, s3p = interp_hfa(age, HFA_BOYS if sex=='L' else HFA_GIRLS)
    if height >= med:
        sd = (s2p - med) / 2
        z  = (height - med) / sd if sd else 0
        if z > 3:
            sd23 = s3p - s2p
            z = 3 + (height - s3p)/sd23 if sd23 else 3
    else:
        sd = (med - s2n) / 2
        z  = (height - med) / sd if sd else 0
        if z < -3:
            sd23 = s2n - s3n
            z = -3 - (s3n - height)/sd23 if sd23 else -3
    return round(z, 2)

# ─────────────────────────────────────────────
# LABEL HELPERS
# ─────────────────────────────────────────────
def tbu_label(z):
    if z < -3: return "Sangat Pendek", "#f87171", "rgba(248,113,113,.14)"
    if z < -2: return "Pendek",        "#fb923c", "rgba(251,146,60,.14)"
    if z <= 3: return "Normal",        "#34d399", "rgba(52,211,153,.14)"
    return     "Tinggi",               "#4fd1c5", "rgba(79,209,197,.14)"

def bbu_label(z):
    if z < -3: return "Sangat Kurang", "#f87171", "rgba(248,113,113,.14)"
    if z < -2: return "Kurang",        "#fb923c", "rgba(251,146,60,.14)"
    if z <=  1: return "Normal",       "#34d399", "rgba(52,211,153,.14)"
    return      "Risiko Lebih",        "#facc15", "rgba(250,204,21,.14)"

def bbtb_label(z):
    if z < -3: return "Gizi Buruk",   "#f87171", "rgba(248,113,113,.14)"
    if z < -2: return "Gizi Kurang",  "#fb923c", "rgba(251,146,60,.14)"
    if z <=  1: return "Gizi Baik",   "#34d399", "rgba(52,211,153,.14)"
    if z <=  2: return "Risiko Lebih","#facc15", "rgba(250,204,21,.14)"
    if z <=  3: return "Gizi Lebih",  "#f97316", "rgba(249,115,22,.14)"
    return      "Obesitas",           "#ef4444", "rgba(239,68,68,.14)"

def risk_tier(pct):
    if pct < 45:
        return dict(cls="rc-normal", icon="✓", title="Tumbuh Kembang Normal",
                    color="#34d399", eyebrow="STATUS PERTUMBUHAN · AMAN",
                    recs=["Pertahankan asupan gizi seimbang dan ASI/MPASI sesuai usia",
                          "Pemantauan rutin di posyandu setiap bulan",
                          "Pastikan jadwal imunisasi lengkap terpenuhi",
                          "Stimulasi tumbuh kembang aktif sesuai usia"])
    elif pct < 65:
        return dict(cls="rc-waspada", icon="○", title="Perlu Diwaspadai",
                    color="#facc15", eyebrow="STATUS PERTUMBUHAN · PERLU PERHATIAN",
                    recs=["Konsultasikan ke petugas gizi atau bidan untuk evaluasi",
                          "Pastikan kecukupan protein hewani, zat besi, dan zinc",
                          "Pantau berat dan tinggi badan setiap bulan",
                          "Perhatikan diversifikasi pola makan anak"])
    elif pct < 80:
        return dict(cls="rc-sedang", icon="⚡", title="Risiko Stunting Sedang",
                    color="#fb923c", eyebrow="PERHATIAN · INTERVENSI GIZI DIANJURKAN",
                    recs=["Segera konsultasikan ke dokter atau ahli gizi anak",
                          "Evaluasi asupan kalori dan protein harian secara mendetail",
                          "Pertimbangkan suplemen: vitamin A, zinc, dan zat besi",
                          "Pemantauan pertumbuhan setiap 2 minggu"])
    else:
        return dict(cls="rc-tinggi", icon="⚠️", title="Risiko Stunting Tinggi",
                    color="#f87171", eyebrow="PERHATIAN · TINDAK LANJUT SEGERA",
                    recs=["Bawa ke puskesmas atau dokter spesialis anak segera",
                          "Diperlukan intervensi gizi intensif dan berkelanjutan",
                          "Evaluasi penyebab: infeksi berulang, MPASI tidak adekuat",
                          "Pemantauan pertumbuhan setiap minggu selama intervensi"])

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(umur_bulan, jk, berat, tinggi, cara_ukur):
    zs_bb_u  = calc_bbu(berat, umur_bulan, jk)
    zs_bb_tb = calc_bbtb(berat, tinggi, jk, cara_ukur, umur_bulan)
    jk_enc   = 1 if jk=='L' else 0
    cara_enc = 1 if cara_ukur=='Berdiri' else 0
    bins = [-1,6,11,23,36,60]; kel = 0
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
    rasio_tb_um=tinggi/umur_bulan if umur_bulan>0 else 0
    rasio_bb_um=berat/umur_bulan if umur_bulan>0 else 0
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
# MODEL LOADER
# ─────────────────────────────────────────────
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights if weights else [1/len(models)]*len(models)
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        for m, w in zip(self.models, self.weights):
            probs += w * m.predict_proba(X)
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
    else:
        model = data; threshold = 0.45
    return model, threshold

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate(umur, berat, tinggi):
    warns = []
    bmin_map = {0:2.0, 6:5.5, 12:7.0, 24:9.0, 36:11.0, 48:13.0, 60:15.0}
    bmax_map = {0:5.5, 6:10.5,12:14.0,24:17.0,36:20.0, 48:23.0, 60:27.0}
    bmin = bmax = None
    for k in sorted(bmin_map):
        if umur >= k: bmin, bmax = bmin_map[k], bmax_map[k]
    if bmin and berat < bmin:
        warns.append(f"BB {berat} kg terlalu rendah untuk usia {umur} bln (min ~{bmin} kg)")
    if bmax and berat > bmax:
        warns.append(f"BB {berat} kg terlalu tinggi untuk usia {umur} bln (maks ~{bmax} kg)")
    tmin_map = {0:44, 6:60, 12:68, 24:78, 36:86, 48:93, 60:100}
    tmax_map = {0:57, 6:74, 12:84, 24:97, 36:107,48:116,60:123}
    tmin = tmax = None
    for k in sorted(tmin_map):
        if umur >= k: tmin, tmax = tmin_map[k], tmax_map[k]
    if tmin and tinggi < tmin:
        warns.append(f"TB {tinggi} cm tidak realistis untuk usia {umur} bln (min ~{tmin} cm)")
    if tmax and tinggi > tmax:
        warns.append(f"TB {tinggi} cm tidak realistis untuk usia {umur} bln (maks ~{tmax} cm)")
    return warns

# ══════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════

# ── INPUT SECTION ──────────────────────────────
st.markdown('<p class="slabel">Data Antropometri Anak</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    umur_bulan = st.number_input("Umur (Bulan)", min_value=0, max_value=60, value=24, step=1)
    berat      = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=35.0,
                                  value=12.0, step=0.1, format="%.1f")
with c2:
    tinggi     = st.number_input("Tinggi / Panjang Badan (cm)", min_value=40.0, max_value=130.0,
                                  value=87.0, step=0.1, format="%.1f")
    cara_ukur  = st.selectbox("Cara Pengukuran",
                               ["Terlentang — Panjang Badan", "Berdiri — Tinggi Badan"],
                               help="Terlentang untuk < 24 bln · Berdiri untuk ≥ 24 bln")

jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

cu_mode = "Berdiri" if cara_ukur.startswith("Berdiri") else "Terlentang"
jk      = "L" if jenis_kelamin == "Laki-laki" else "P"

# Warnings
if umur_bulan < 24 and cu_mode == "Berdiri":
    st.warning("⚠️ Untuk usia < 24 bulan, disarankan pengukuran Terlentang.")
elif umur_bulan >= 24 and cu_mode == "Terlentang":
    st.warning("⚠️ Untuk usia ≥ 24 bulan, disarankan pengukuran Berdiri.")
for w in validate(umur_bulan, berat, tinggi):
    st.warning(f"⚠️ {w}")

st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

# ── PREDICT BUTTON ─────────────────────────────
btn = st.button("Analisis & Prediksi Sekarang")

# ── RESULTS ────────────────────────────────────
if btn:
    with st.spinner("Menganalisis data antropometri..."):
        try:
            model, threshold = load_model()
            X, zs_bbu, zs_bbtb = build_features(umur_bulan, jk, berat, tinggi, cu_mode)
            proba  = model.predict_proba(X)[0]
            pct    = proba[1] * 100
            zs_tbu = calc_tbu(tinggi, umur_bulan, jk)
        except Exception as e:
            st.error(f"Error saat prediksi: {e}"); st.stop()

    tier = risk_tier(pct)

    st.markdown('<p class="slabel" style="margin-top:1.2rem;">Hasil Analisis</p>',
                unsafe_allow_html=True)

    # ── STATUS CARD
    bar_pos = max(2, min(97, pct))
    st.markdown(f"""
    <div class="rcard {tier['cls']}">
        <div class="rc-eyebrow" style="color:{tier['color']};">{tier['eyebrow']}</div>
        <div class="rc-check">{tier['icon']}</div>
        <div class="rc-title" style="color:{tier['color']};">{tier['title']}</div>
        <div class="rc-prob">
            Probabilitas stunting &nbsp;—&nbsp;
            <strong style="color:{tier['color']};font-size:1rem;">{pct:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── GAUGE BAR
    st.markdown(f"""
    <div class="gauge">
        <div class="gtop">
            <span class="gttl">Indeks Probabilitas Stunting</span>
            <span class="gnum" style="color:{tier['color']};">
                {pct:.1f}<span style="font-size:.82rem;opacity:.5;">%</span>
            </span>
        </div>
        <div class="gtrack">
            <div class="gfill"></div>
            <div class="gpip" style="left:{bar_pos}%;color:{tier['color']};background:{tier['color']};"></div>
        </div>
        <div class="gtick">
            <span>Normal</span><span>Batas 45%</span><span>Risiko Tinggi</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Z-SCORE CARDS  (BB/U · BB/TB · TB/U)
    bc,  bcol,  bbg   = bbu_label(zs_bbu)
    btbc,btbcol,btbbg = bbtb_label(zs_bbtb)
    tc,  tcol,  tbg   = tbu_label(zs_tbu)

    st.markdown(f"""
    <div class="zrow">
        <div class="zcard">
            <div class="ztag">Z-Score BB / Umur</div>
            <div class="zval" style="color:{bcol};">{zs_bbu:+.2f}</div>
            <div class="zdesc">Berat Badan menurut Umur</div>
            <span class="zbadge" style="background:{bbg};color:{bcol};">{bc}</span>
        </div>
        <div class="zcard">
            <div class="ztag">Z-Score BB / Tinggi</div>
            <div class="zval" style="color:{btbcol};">{zs_bbtb:+.2f}</div>
            <div class="zdesc">Berat Badan menurut Tinggi</div>
            <span class="zbadge" style="background:{btbbg};color:{btbcol};">{btbc}</span>
        </div>
        <div class="zcard" style="{'border-color:rgba(248,113,113,.35);' if zs_tbu < -2 else ''}">
            <div class="ztag">Z-Score TB / Umur ★</div>
            <div class="zval" style="color:{tcol};">{zs_tbu:+.2f}</div>
            <div class="zdesc">Indikator Stunting Utama</div>
            <span class="zbadge" style="background:{tbg};color:{tcol};">{tc}</span>
        </div>
    </div>
    <div style="font-size:.68rem;color:#475569;text-align:right;margin-top:-.4rem;margin-bottom:.7rem;">
        ★ TB/U = indikator stunting per WHO &amp; Permenkes No. 2/2020
    </div>
    """, unsafe_allow_html=True)

    # ── REKOMENDASI
    recs_html = "".join([f'<div class="rec-item">{r}</div>' for r in tier["recs"]])
    st.markdown(f"""
    <div class="recbox" style="border-left-color:{tier['color']};">
        <strong style="color:{tier['color']};">Rekomendasi Intervensi</strong>
        <div style="margin-top:.45rem;">{recs_html}</div>
        <div class="rec-disc">
            Skrining AI berbasis WHO 2006 &amp; Permenkes No. 2/2020.
            Tidak menggantikan diagnosis medis. Konsultasikan ke tenaga kesehatan.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.7rem;line-height:1.9;
            padding-top:1rem;margin-top:1.5rem;border-top:1px solid rgba(255,255,255,.06);">
    WHO 2006 Multicentre Growth Reference · Permenkes No. 2 Tahun 2020<br>
    Ensemble CatBoost + XGBoost · Skrining Awal — Bukan Pengganti Diagnosis Medis
</div>
""", unsafe_allow_html=True)
