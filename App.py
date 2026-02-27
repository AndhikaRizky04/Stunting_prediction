import streamlit as st
import pandas as pd
import numpy as np
import joblib
import bisect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Prediksi Stunting Balita",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0b1120;
    --card:   #111827;
    --card2:  #1a2535;
    --card3:  #1f2d42;
    --gold:   #c9a84c;
    --gold-lt:#e8d48b;
    --gold-dk:#9a6e1a;
    --gold-dim:rgba(201,168,76,.14);
    --t1:#f1f5f9; --t2:#94a3b8; --t3:#64748b; --t4:#475569;
    --border:rgba(255,255,255,.07);
    --bg:rgba(201,168,76,.25);
    --green:#34d399; --yellow:#facc15; --orange:#fb923c; --red:#f87171;
    --teal:#4fd1c5;
}

html,body,[class*="css"],.stApp,.main,
div[data-testid="stAppViewContainer"],
div[data-testid="stMain"]{
    font-family:'DM Sans',sans-serif!important;
    background-color:var(--navy)!important;
    color:var(--t1)!important;
}
#MainMenu,footer,header{visibility:hidden!important;}
.block-container{padding:0 1.5rem 4rem!important;max-width:860px!important;}

/* ─── HERO ─────────────────────────────────── */
.hero{
    position:relative;text-align:center;
    padding:2.8rem 2rem 2.5rem;margin-bottom:0;
    background:linear-gradient(160deg,#0f1e32 0%,#0b1120 60%,#101a2e 100%);
    border:1px solid rgba(201,168,76,.22);
    border-radius:20px;overflow:hidden;
    animation:fadeDown .7s ease both;
}
.hero::before{
    content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 55% at 50% -5%,rgba(201,168,76,.10) 0%,transparent 70%);
}
.hero::after{
    content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(201,168,76,.5),transparent);
}
/* subtle grid texture */
.hero-grid{
    position:absolute;inset:0;pointer-events:none;
    background-image:
        linear-gradient(rgba(201,168,76,.04) 1px,transparent 1px),
        linear-gradient(90deg,rgba(201,168,76,.04) 1px,transparent 1px);
    background-size:40px 40px;
}
.hero-badge{
    display:inline-flex;align-items:center;gap:.45rem;
    background:rgba(201,168,76,.12);border:1px solid rgba(201,168,76,.28);
    color:var(--gold-lt);font-size:.67rem;font-weight:600;
    letter-spacing:.16em;text-transform:uppercase;
    padding:.32rem 1rem;border-radius:100px;
    margin-bottom:1.4rem;position:relative;
}
.hero-title{
    font-family:'Cormorant Garamond',serif!important;
    font-size:3rem!important;font-weight:300!important;
    letter-spacing:-.02em!important;line-height:1.05!important;
    color:var(--t1)!important;margin:0 0 .5rem!important;position:relative;
}
.hero-title em{font-style:italic;color:var(--gold-lt);}
.hero-rule{
    width:40px;height:1px;margin:.8rem auto;
    background:linear-gradient(90deg,transparent,var(--gold),transparent);
    position:relative;
}
.hero-sub{
    font-size:.82rem!important;color:var(--t3)!important;
    font-weight:300!important;line-height:1.7!important;
    margin:0!important;position:relative;letter-spacing:.01em;
}

/* ─── TABS ──────────────────────────────────── */
div[data-testid="stTabs"] [role="tablist"]{
    gap:.25rem;
    border-bottom:1px solid rgba(201,168,76,.2)!important;
    margin-bottom:1.5rem;
}
div[data-testid="stTabs"] [role="tab"]{
    background:transparent!important;border:none!important;
    color:var(--t3)!important;font-family:'DM Sans',sans-serif!important;
    font-size:.82rem!important;font-weight:500!important;
    padding:.55rem 1.1rem!important;border-radius:8px 8px 0 0!important;
    letter-spacing:.02em!important;transition:color .15s!important;
}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
    background:var(--gold-dim)!important;color:var(--gold-lt)!important;
    border-bottom:2px solid var(--gold)!important;
}
div[data-testid="stTabs"] [role="tab"]:hover{
    color:var(--t1)!important;background:rgba(255,255,255,.04)!important;
}

/* ─── SECTION LABEL ─────────────────────────── */
.slabel{
    font-size:.61rem;font-weight:600;letter-spacing:.22em;text-transform:uppercase;
    color:var(--gold);display:flex;align-items:center;gap:.7rem;
    margin:1.4rem 0 1rem;
}
.slabel::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(201,168,76,.3),transparent);}

/* ─── INPUTS ────────────────────────────────── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"]>div>div{
    background:var(--card)!important;
    border:1.5px solid rgba(255,255,255,.09)!important;
    border-radius:10px!important;color:var(--t1)!important;
    font-family:'DM Sans',sans-serif!important;font-size:.93rem!important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"]>div>div:focus-within{
    border-color:var(--gold)!important;
    box-shadow:0 0 0 3px rgba(201,168,76,.12)!important;
}
div[data-testid="stSelectbox"] ul{
    background:var(--card2)!important;border:1px solid rgba(201,168,76,.25)!important;
    border-radius:10px!important;padding:.35rem!important;
}
div[data-testid="stSelectbox"] li{border-radius:7px!important;color:var(--t2)!important;}
div[data-testid="stSelectbox"] li:hover{background:var(--gold-dim)!important;color:var(--gold-lt)!important;}
label[data-testid="stWidgetLabel"] p,div[data-testid="stWidgetLabel"] p{
    color:var(--t3)!important;font-size:.74rem!important;
    font-weight:600!important;letter-spacing:.13em!important;text-transform:uppercase!important;
}
div[data-testid="stNumberInput"] button{
    background:var(--card2)!important;border:1px solid var(--border)!important;
    color:var(--t3)!important;border-radius:7px!important;
}
div[data-testid="stSlider"] [data-testid="stSlider"]{background:var(--gold)!important;}

/* ─── BUTTON ────────────────────────────────── */
.stButton>button{
    width:100%!important;
    background:linear-gradient(135deg,var(--gold-dk),var(--gold) 50%,var(--gold-lt))!important;
    color:#0b1120!important;border:none!important;border-radius:12px!important;
    padding:.9rem 2rem!important;font-size:.88rem!important;font-weight:600!important;
    letter-spacing:.09em!important;text-transform:uppercase!important;
    font-family:'DM Sans',sans-serif!important;
    box-shadow:0 6px 20px rgba(201,168,76,.26)!important;
    transition:all .25s ease!important;
}
.stButton>button:hover{
    transform:translateY(-2px)!important;
    box-shadow:0 10px 28px rgba(201,168,76,.36)!important;
}

/* ─── RESULT CARD ───────────────────────────── */
.rcard{
    border-radius:18px;padding:2.2rem 1.8rem;text-align:center;
    position:relative;overflow:hidden;margin-bottom:1.2rem;
    animation:pop .5s cubic-bezier(.34,1.4,.64,1) both;
}
.rcard::before{
    content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 70% 50% at 50% 0%,var(--glow),transparent 70%);
}
.rc-n{background:linear-gradient(160deg,#091a12,#0d2218);border:1px solid rgba(52,211,153,.25);--glow:rgba(52,211,153,.13);}
.rc-w{background:linear-gradient(160deg,#1a1400,#221a00);border:1px solid rgba(250,204,21,.25);--glow:rgba(250,204,21,.11);}
.rc-s{background:linear-gradient(160deg,#1a0e00,#221400);border:1px solid rgba(251,146,60,.25);--glow:rgba(251,146,60,.12);}
.rc-t{background:linear-gradient(160deg,#1a0909,#220d0d);border:1px solid rgba(248,113,113,.25);--glow:rgba(248,113,113,.13);}
.rc-eye{font-size:.61rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;opacity:.65;margin-bottom:.65rem;position:relative;}
.rc-ico{font-size:2.1rem;margin-bottom:.2rem;position:relative;line-height:1;}
.rc-ttl{font-family:'Cormorant Garamond',serif;font-size:2.35rem;font-weight:400;letter-spacing:-.02em;line-height:1;margin-bottom:.4rem;position:relative;}
.rc-sub{font-size:.82rem;opacity:.6;position:relative;}

/* ─── CONFIDENCE RING ───────────────────────── */
.conf-wrap{
    background:var(--card);border:1px solid var(--border);
    border-radius:14px;padding:1.1rem;margin-bottom:1rem;
    display:flex;align-items:center;gap:1.1rem;
}
.conf-ring{
    width:62px;height:62px;border-radius:50%;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    font-family:'Cormorant Garamond',serif;font-size:1.1rem;font-weight:600;
    background:conic-gradient(var(--rc) var(--pct),rgba(255,255,255,.06) 0);
    position:relative;
}
.conf-ring::before{
    content:'';position:absolute;inset:8px;border-radius:50%;
    background:var(--card);
}
.conf-ring span{position:relative;z-index:1;}
.conf-info{flex:1;}
.conf-lbl{font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;color:var(--t3);font-weight:600;margin-bottom:.3rem;}
.conf-val{font-family:'Cormorant Garamond',serif;font-size:1.55rem;font-weight:300;line-height:1;margin-bottom:.2rem;}
.conf-desc{font-size:.75rem;color:var(--t3);}

/* ─── GAUGE ─────────────────────────────────── */
.gauge{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem;}
.gtop{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.75rem;}
.gttl{font-size:.61rem;letter-spacing:.15em;text-transform:uppercase;font-weight:600;color:var(--t3);}
.gnum{font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:300;}
.gtrack{height:6px;border-radius:100px;background:rgba(255,255,255,.07);position:relative;}
.gfill{position:absolute;left:0;top:0;height:100%;border-radius:100px;background:linear-gradient(90deg,#34d399 0%,#facc15 45%,#fb923c 70%,#f87171 100%);width:100%;}
.gpip{position:absolute;top:50%;transform:translate(-50%,-50%);width:12px;height:12px;border-radius:50%;border:2px solid var(--navy);z-index:2;box-shadow:0 0 8px currentColor;}
.gtick{display:flex;justify-content:space-between;margin-top:.4rem;font-size:.63rem;color:var(--t4);}

/* ─── Z-SCORE CARDS ─────────────────────────── */
.zrow{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:.5rem;}
.zcard{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.2rem .9rem;text-align:center;transition:border-color .2s,transform .15s;}
.zcard:hover{border-color:rgba(201,168,76,.3);transform:translateY(-2px);}
.ztag{font-size:.59rem;letter-spacing:.12em;text-transform:uppercase;font-weight:600;color:var(--t3);margin-bottom:.45rem;}
.zval{font-family:'Cormorant Garamond',serif;font-size:2.5rem;font-weight:300;line-height:1;margin-bottom:.3rem;}
.zdesc{font-size:.67rem;color:var(--t3);margin-bottom:.4rem;}
.zbadge{font-size:.69rem;font-weight:500;padding:.2rem .6rem;border-radius:100px;display:inline-block;}

/* ─── INFO BAND ─────────────────────────────── */
.iband{
    display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;
    margin-bottom:1rem;
}
.ibox{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.9rem .7rem;text-align:center;transition:border-color .2s;}
.ibox:hover{border-color:rgba(201,168,76,.25);}
.ibox-lbl{font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--t3);font-weight:600;margin-bottom:.3rem;}
.ibox-val{font-family:'Cormorant Garamond',serif;font-size:1.25rem;font-weight:300;line-height:1;margin-bottom:.15rem;}
.ibox-sub{font-size:.63rem;color:var(--t4);}

/* ─── RECOMMENDATION ───────────────────────── */
.recbox{background:var(--card);border:1px solid var(--border);border-left:3px solid;border-radius:14px;padding:1.1rem 1.3rem;font-size:.85rem;line-height:1.75;color:var(--t2);margin-bottom:1rem;}
.rec-head{font-size:.61rem;letter-spacing:.15em;text-transform:uppercase;font-weight:600;margin-bottom:.75rem;}
.rec-item{display:flex;align-items:flex-start;gap:.5rem;margin:.25rem 0;font-size:.86rem;}
.rec-item::before{content:'→';color:var(--gold);flex-shrink:0;}
.rec-disc{font-size:.71rem;color:var(--t4);margin-top:.6rem;font-style:italic;border-top:1px solid var(--border);padding-top:.6rem;}

/* ─── FUTURE TABLE ──────────────────────────── */
.ftable{width:100%;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;border:1px solid var(--border);margin-top:.8rem;}
.ftable thead tr{background:var(--card3);}
.ftable th{padding:.6rem .9rem;font-size:.67rem;color:var(--t3);font-weight:600;letter-spacing:.1em;text-transform:uppercase;text-align:left;}
.ftable td{padding:.65rem .9rem;font-size:.83rem;color:var(--t2);border-top:1px solid var(--border);}
.ftable tr:hover td{background:rgba(255,255,255,.02);}

/* ─── ABOUT / METRIC CARDS ──────────────────── */
.mrow{display:grid;grid-template-columns:repeat(5,1fr);gap:.7rem;margin-bottom:1.4rem;}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem .8rem;text-align:center;}
.mcard-val{font-family:'Cormorant Garamond',serif;font-size:1.8rem;font-weight:300;line-height:1;color:var(--gold-lt);margin-bottom:.25rem;}
.mcard-lbl{font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);}

/* ─── INTERPRETATION BOX ────────────────────── */
.interpbox{background:var(--card);border:1px solid var(--border);border-left:3px solid rgba(201,168,76,.5);border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem;}
.interp-ttl{font-size:.61rem;letter-spacing:.15em;text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:.7rem;}
.interp-row{display:flex;align-items:center;gap:.65rem;padding:.38rem .5rem;border-radius:8px;margin-bottom:.2rem;font-size:.84rem;}
.interp-row.active{border:1px solid;}
.interp-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}

/* ─── ALERTS ────────────────────────────────── */
div[data-testid="stAlert"]{background:rgba(201,168,76,.07)!important;border:1px solid rgba(201,168,76,.25)!important;border-radius:10px!important;color:var(--gold-lt)!important;}
div[data-testid="stSpinner"] p{color:var(--t2)!important;}

/* ─── ANIMATIONS ────────────────────────────── */
@keyframes fadeDown{from{opacity:0;transform:translateY(-16px);}to{opacity:1;transform:translateY(0);}}
@keyframes pop{from{opacity:0;transform:scale(.94);}to{opacity:1;transform:scale(1);}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# WHO TABLES
# ══════════════════════════════════════════════
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

# ══════════════════════════════════════════════
# Z-SCORE FUNCTIONS
# ══════════════════════════════════════════════
def lms_z(X, L, M, S):
    z = ((X/M)**L - 1)/(L*S) if L != 0 else np.log(X/M)/S
    if z > 3:
        SD3 = M*(1+L*S*3)**(1/L); SD23 = SD3 - M*(1+L*S*2)**(1/L)
        z = 3 + (X-SD3)/SD23 if SD23 else 3
    elif z < -3:
        SD3 = M*(1+L*S*(-3))**(1/L); SD23 = M*(1+L*S*(-2))**(1/L) - SD3
        z = -3 + (X-SD3)/SD23 if SD23 else -3
    return round(z, 2)

def _ilms_age(age, tbl):
    ages=[r[0] for r in tbl]; a=max(0,min(60,int(round(age))))
    idx=bisect.bisect_left(ages,a)
    if idx>=len(tbl): idx=len(tbl)-1
    elif idx>0 and ages[idx]!=a:
        a0,a1=ages[idx-1],ages[idx]; t=(a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(tbl[idx-1][i]+t*(tbl[idx][i]-tbl[idx-1][i]) for i in range(1,4))
    return tbl[idx][1],tbl[idx][2],tbl[idx][3]

def _ilms_h(h, tbl):
    hs=[r[0] for r in tbl]; h=max(hs[0],min(hs[-1],h))
    idx=bisect.bisect_left(hs,h)
    if idx>=len(tbl): idx=len(tbl)-1
    elif idx>0:
        h0,h1=hs[idx-1],hs[idx]; t=(h-h0)/(h1-h0) if h1!=h0 else 0
        return tuple(tbl[idx-1][i]+t*(tbl[idx][i]-tbl[idx-1][i]) for i in range(1,4))
    return tbl[idx][1],tbl[idx][2],tbl[idx][3]

def _ihfa(age, tbl):
    ages=[r[0] for r in tbl]; a=max(0,min(60,age))
    idx=bisect.bisect_left(ages,a)
    if idx>=len(tbl): idx=len(tbl)-1
    elif idx>0 and ages[idx]!=a:
        a0,a1=ages[idx-1],ages[idx]; t=(a-a0)/(a1-a0) if a1!=a0 else 0
        return tuple(tbl[idx-1][i]+t*(tbl[idx][i]-tbl[idx-1][i]) for i in range(1,6))
    return tuple(tbl[idx][1:6])

def calc_bbu(w, age, sex):
    L,M,S=_ilms_age(age, WFA_BOYS if sex=='L' else WFA_GIRLS)
    return lms_z(w,L,M,S)

def calc_bbtb(w, h, sex, cara, age):
    hh=h
    if cara=='Terlentang' and age>=24: hh=h-0.7
    elif cara=='Berdiri' and age<24: hh=h+0.7
    L,M,S=_ilms_h(hh, WFL_BOYS if sex=='L' else WFL_GIRLS)
    return lms_z(w,L,M,S)

def calc_tbu(h, age, sex):
    s3n,s2n,med,s2p,s3p=_ihfa(age, HFA_BOYS if sex=='L' else HFA_GIRLS)
    if h>=med:
        sd=(s2p-med)/2; z=(h-med)/sd if sd else 0
        if z>3: sd23=s3p-s2p; z=3+(h-s3p)/sd23 if sd23 else 3
    else:
        sd=(med-s2n)/2; z=(h-med)/sd if sd else 0
        if z<-3: sd23=s2n-s3n; z=-3-(s3n-h)/sd23 if sd23 else -3
    return round(z,2)

def get_median_tbu(age, sex):
    _,_,med,_,_=_ihfa(age, HFA_BOYS if sex=='L' else HFA_GIRLS)
    return round(med,1)

def get_median_bbu(age, sex):
    _,M,_=_ilms_age(age, WFA_BOYS if sex=='L' else WFA_GIRLS)
    return round(M,2)

# ══════════════════════════════════════════════
# LABEL HELPERS
# ══════════════════════════════════════════════
def lbl_bbu(z):
    if z<-3: return "Sangat Kurang","#f87171","rgba(248,113,113,.14)"
    if z<-2: return "Kurang",       "#fb923c","rgba(251,146,60,.14)"
    if z<=1: return "Normal",       "#34d399","rgba(52,211,153,.14)"
    return        "Risiko Lebih",   "#facc15","rgba(250,204,21,.14)"

def lbl_bbtb(z):
    if z<-3: return "Gizi Buruk",  "#f87171","rgba(248,113,113,.14)"
    if z<-2: return "Gizi Kurang", "#fb923c","rgba(251,146,60,.14)"
    if z<=1: return "Gizi Baik",   "#34d399","rgba(52,211,153,.14)"
    if z<=2: return "Risiko Lebih","#facc15","rgba(250,204,21,.14)"
    if z<=3: return "Gizi Lebih",  "#f97316","rgba(249,115,22,.14)"
    return "Obesitas",             "#ef4444","rgba(239,68,68,.14)"

def lbl_tbu(z):
    if z<-3: return "Sangat Pendek","#f87171","rgba(248,113,113,.14)"
    if z<-2: return "Pendek",       "#fb923c","rgba(251,146,60,.14)"
    if z<=3: return "Normal",       "#34d399","rgba(52,211,153,.14)"
    return "Tinggi",                "#4fd1c5","rgba(79,209,197,.14)"

def risk_tier(pct):
    if pct < 45:
        return dict(cls="rc-n",icon="✓",title="Tumbuh Kembang Normal",color="#34d399",
                    eyebrow="STATUS PERTUMBUHAN · AMAN",
                    level="Rendah",priority="Rutin",
                    recs_immediate=["Pertahankan asupan gizi seimbang dan ASI/MPASI sesuai usia",
                                    "Pemantauan rutin di posyandu setiap bulan"],
                    recs_nutrition=["Konsumsi protein hewani: telur, ikan, daging, ayam",
                                    "Lengkapi dengan sayuran berwarna dan buah-buahan"],
                    recs_monitoring=["Timbang dan ukur tinggi badan setiap bulan",
                                     "Catat di buku KIA dan bandingkan dengan kurva WHO"])
    elif pct < 65:
        return dict(cls="rc-w",icon="○",title="Perlu Diwaspadai",color="#facc15",
                    eyebrow="STATUS PERTUMBUHAN · PERLU PERHATIAN",
                    level="Sedang-Rendah",priority="Segera",
                    recs_immediate=["Konsultasikan ke petugas gizi atau bidan untuk evaluasi",
                                    "Evaluasi kecukupan makan harian anak secara mendetail"],
                    recs_nutrition=["Tingkatkan asupan protein: 2–3 porsi/hari dari sumber hewani",
                                    "Pastikan asupan zat besi dan zinc terpenuhi dari makanan"],
                    recs_monitoring=["Pantau BB dan TB setiap bulan dengan cermat",
                                     "Catat pertumbuhan di grafik KMS dan waspadai plateau"])
    elif pct < 80:
        return dict(cls="rc-s",icon="⚡",title="Risiko Stunting Sedang",color="#fb923c",
                    eyebrow="PERHATIAN · INTERVENSI GIZI DIANJURKAN",
                    level="Sedang-Tinggi",priority="Mendesak",
                    recs_immediate=["Bawa ke dokter atau ahli gizi anak dalam waktu dekat",
                                    "Periksa kemungkinan penyakit infeksi berulang (ISPA, diare)"],
                    recs_nutrition=["Program PMT (Pemberian Makanan Tambahan) jika tersedia",
                                    "Pertimbangkan suplementasi: vitamin A, zinc, zat besi dosis terapeutik"],
                    recs_monitoring=["Pemantauan pertumbuhan setiap 2 minggu",
                                     "Foto dokumentasi untuk melacak perubahan visual kondisi anak"])
    else:
        return dict(cls="rc-t",icon="⚠️",title="Risiko Stunting Tinggi",color="#f87171",
                    eyebrow="PERHATIAN · TINDAK LANJUT SEGERA",
                    level="Tinggi",priority="Darurat",
                    recs_immediate=["Bawa ke puskesmas atau dokter spesialis anak SEGERA",
                                    "Lakukan pemeriksaan laboratorium: Hb, albumin, infeksi"],
                    recs_nutrition=["Intervensi gizi intensif terstruktur dengan pendampingan nakes",
                                    "F-75 / F-100 atau RUTF jika dinyatakan SAM oleh dokter"],
                    recs_monitoring=["Pemantauan pertumbuhan setiap minggu dengan tim medis",
                                     "Rawat inap mungkin diperlukan jika ada komplikasi"])

# ══════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════
def build_features(umur_bulan, jk, berat, tinggi, cara_ukur):
    zs_bb_u  = calc_bbu(berat, umur_bulan, jk)
    zs_bb_tb = calc_bbtb(berat, tinggi, jk, cara_ukur, umur_bulan)
    jk_enc   = 1 if jk=='L' else 0
    cara_enc = 1 if cara_ukur=='Berdiri' else 0
    bins=[-1,6,11,23,36,60]; kel=0
    for i in range(len(bins)-1):
        if umur_bulan>bins[i] and umur_bulan<=bins[i+1]: kel=[0,1,2,3,4][i]
    fw=1 if umur_bulan<=23 else 0; fm=1 if 6<=umur_bulan<=23 else 0
    fb=1 if umur_bulan<=23 else 0; asq=umur_bulan**2; alg=np.log1p(umur_bulan)
    bsk=1 if zs_bb_u<-3 else 0; bk=1 if -3<=zs_bb_u<-2 else 0
    bn=1 if -2<=zs_bb_u<=1 else 0; brl=1 if zs_bb_u>1 else 0; uw=1 if zs_bb_u<-2 else 0
    gb=1 if zs_bb_tb<-3 else 0; gk=1 if -3<=zs_bb_tb<-2 else 0
    gbk=1 if -2<=zs_bb_tb<=1 else 0; rgl=1 if 1<zs_bb_tb<=2 else 0
    gl=1 if 2<zs_bb_tb<=3 else 0; ob=1 if zs_bb_tb>3 else 0; wst=1 if zs_bb_tb<-2 else 0
    bm=1 if -2.5<=zs_bb_u<-2 else 0; btm=1 if -2.5<=zs_bb_tb<-2 else 0
    pb2=max(0,zs_bb_u-(-2)); ptb2=max(0,zs_bb_tb-(-2))
    pb3=max(0,zs_bb_u-(-3)); ptb3=max(0,zs_bb_tb-(-3))
    fdbl=1 if uw==1 and wst==1 else 0; fds=1 if bsk==1 and gb==1 else 0
    fas=1 if bsk==1 or gb==1 else 0
    sk=bsk*2+bk+gb*2+gk; ji=(1 if uw else 0)+(1 if wst else 0)
    az=(zs_bb_u+zs_bb_tb)/2; mnz=min(zs_bb_u,zs_bb_tb); mxz=max(zs_bb_u,zs_bb_tb)
    gp=zs_bb_u-zs_bb_tb; abg=abs(gp); pr=zs_bb_u*zs_bb_tb
    hm=(2*zs_bb_u*zs_bb_tb/(zs_bb_u+zs_bb_tb)) if (zs_bb_u!=0 and zs_bb_tb!=0) else 0.0
    bs2=zs_bb_u**2; bts2=zs_bb_tb**2; bs3=zs_bb_u**3; bts3=zs_bb_tb**3
    uxr=umur_bulan*sk; uxm=umur_bulan*mnz; uxb=umur_bulan*zs_bb_u; uxt=umur_bulan*zs_bb_tb
    bxw=fb*wst; mxu=fm*uw; jxm=jk_enc*mnz; jxb=jk_enc*zs_bb_u; jxr=jk_enc*sk
    bmi=berat/(tinggi/100)**2; rbt=berat/tinggi
    rtu=tinggi/umur_bulan if umur_bulan>0 else 0; rbu=berat/umur_bulan if umur_bulan>0 else 0
    rb=float(1/(1+np.exp(-zs_bb_u))); rbt2=float(1/(1+np.exp(-zs_bb_tb)))
    row={
        'umur_bulan':umur_bulan,'jk_encoded':jk_enc,'cara_ukur_encoded':cara_enc,
        'kel_usia_permenkes':float(kel),'f_window_1000hpk':fw,'f_masa_mpasi':fm,
        'f_baduta':fb,'age_sq':asq,'age_log':alg,'zs_bb_u':zs_bb_u,
        'f_bb_sangat_kurang':bsk,'f_bb_kurang':bk,'f_bb_normal':bn,'f_bb_risiko_lebih':brl,
        'f_underweight':uw,'zs_bb_tb':zs_bb_tb,'f_gizi_buruk':gb,'f_gizi_kurang':gk,
        'f_gizi_baik':gbk,'f_risiko_gizi_lebih':rgl,'f_gizi_lebih':gl,'f_obesitas':ob,
        'f_wasting':wst,'f_bb_mild':bm,'f_bbtb_mild':btm,
        'prox_bbu_ke_minus2':pb2,'prox_bbtb_ke_minus2':ptb2,'prox_bbu_ke_minus3':pb3,
        'prox_bbtb_ke_minus3':ptb3,'f_double_malnutrisi':fdbl,'f_double_severe':fds,
        'f_any_severe':fas,'skor_risiko_gizi':sk,'jml_indeks_masalah':ji,
        'avg_zs_bbu_bbtb':az,'min_zs_bbu_bbtb':mnz,'max_zs_bbu_bbtb':mxz,
        'gap_bbu_bbtb':gp,'abs_gap_bbu_bbtb':abg,'product_zs':pr,'harmonic_zs':hm,
        'zs_bbu_sq':bs2,'zs_bbtb_sq':bts2,'zs_bbu_cb':bs3,'zs_bbtb_cb':bts3,
        'umur_x_risiko':uxr,'umur_x_minzs':uxm,'umur_x_bbu':uxb,'umur_x_bbtb':uxt,
        'baduta_x_wasting':bxw,'mpasi_x_underw':mxu,'jk_x_minzs':jxm,'jk_x_bbu':jxb,
        'jk_x_risiko':jxr,'bmi_proxy':bmi,'rasio_bb_tb':rbt,'rasio_tb_umur':rtu,
        'rasio_bb_umur':rbu,'berat':berat,'tinggi':tinggi,'rank_bbu':rb,'rank_bbtb':rbt2
    }
    feats=['umur_bulan','jk_encoded','cara_ukur_encoded','kel_usia_permenkes',
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
           'rank_bbu','rank_bbtb']
    df=pd.DataFrame([row])[feats]
    return df.replace([np.inf,-np.inf],np.nan).fillna(0), zs_bb_u, zs_bb_tb

# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models=models; self.weights=weights if weights else [1/len(models)]*len(models)
    def predict_proba(self, X):
        p=np.zeros((X.shape[0],2))
        for m,w in zip(self.models,self.weights): p+=w*m.predict_proba(X)
        return p

@st.cache_resource
def load_model():
    import sys,types
    mod=types.ModuleType("__main__"); mod.EnsembleModel=EnsembleModel
    sys.modules["__main__"]=mod
    data=joblib.load("model.pkl")
    if isinstance(data,dict): model=data["model"]; threshold=float(data.get("threshold",0.45))
    else: model=data; threshold=0.45
    return model,threshold

def predict(umur,jk,berat,tinggi,cara):
    model,_=load_model()
    X,zb,zbt=build_features(umur,jk,berat,tinggi,cara)
    p=model.predict_proba(X)[0]
    return p[1]*100, max(p)*100, zb, zbt

# ══════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════
def validate(umur, berat, tinggi):
    warns=[]
    bmin_m={0:2.0,6:5.5,12:7.0,24:9.0,36:11.0,48:13.0,60:15.0}
    bmax_m={0:5.5,6:10.5,12:14.0,24:17.0,36:20.0,48:23.0,60:27.0}
    bmin=bmax=None
    for k in sorted(bmin_m):
        if umur>=k: bmin,bmax=bmin_m[k],bmax_m[k]
    if bmin and berat<bmin: warns.append(f"BB {berat} kg terlalu rendah untuk usia {umur} bln (min ≈{bmin} kg)")
    if bmax and berat>bmax: warns.append(f"BB {berat} kg terlalu tinggi untuk usia {umur} bln (maks ≈{bmax} kg)")
    tmin_m={0:44,6:60,12:68,24:78,36:86,48:93,60:100}
    tmax_m={0:57,6:74,12:84,24:97,36:107,48:116,60:123}
    tmin=tmax=None
    for k in sorted(tmin_m):
        if umur>=k: tmin,tmax=tmin_m[k],tmax_m[k]
    if tmin and tinggi<tmin: warns.append(f"TB {tinggi} cm tidak realistis untuk usia {umur} bln (min ≈{tmin} cm)")
    if tmax and tinggi>tmax: warns.append(f"TB {tinggi} cm tidak realistis untuk usia {umur} bln (maks ≈{tmax} cm)")
    return warns

# ══════════════════════════════════════════════
# FUTURE PROJECTION
# ══════════════════════════════════════════════
def project_future(umur, jk, berat, tinggi, cara, months_ahead=6):
    """Project probable growth in next N months using WHO median velocity."""
    model, _ = load_model()
    results = []
    # WHO approximate monthly weight gain (kg) and height gain (cm) by age group
    weight_velocity = {0:0.90, 3:0.65, 6:0.40, 9:0.32, 12:0.22, 18:0.18, 24:0.14, 36:0.12, 48:0.10}
    height_velocity = {0:3.5,  3:2.5,  6:1.8,  9:1.4,  12:1.1,  18:1.0,  24:0.9,  36:0.8,  48:0.7}

    def get_vel(d, age):
        keys = sorted(d.keys())
        for k in reversed(keys):
            if age >= k: return d[k]
        return list(d.values())[-1]

    cur_b, cur_t, cur_u = berat, tinggi, umur
    for m in range(1, months_ahead+1):
        wv = get_vel(weight_velocity, cur_u)
        hv = get_vel(height_velocity, cur_u)
        cur_b = round(cur_b + wv, 2)
        cur_t = round(cur_t + hv, 1)
        cur_u = min(60, umur + m)
        try:
            X, zb, zbt = build_features(cur_u, jk, cur_b, cur_t, cara)
            prob = model.predict_proba(X)[0][1] * 100
            zt   = calc_tbu(cur_t, cur_u, jk)
            tc, tcol, _ = lbl_tbu(zt)
        except:
            prob, zt, tc, tcol = 0, 0, "—", "#64748b"
        results.append({
            "bulan_ke": m, "umur": cur_u, "bb": cur_b, "tb": cur_t,
            "zs_tbu": zt, "status_tb": tc, "tcol": tcol, "prob": prob,
        })
    return results

# ══════════════════════════════════════════════
# PLOT — FUTURE PROB TREND
# ══════════════════════════════════════════════
def plot_future(base_pct, rows, tier_color):
    months = [0] + [r["bulan_ke"] for r in rows]
    probs  = [base_pct] + [r["prob"] for r in rows]
    tbus   = [None] + [r["zs_tbu"] for r in rows]

    fig, ax = plt.subplots(figsize=(9,3.4))
    fig.patch.set_facecolor("#111827"); ax.set_facecolor("#0b1120")
    ax.fill_between(months, probs, alpha=.12,
                    color="#34d399" if base_pct<45 else ("#facc15" if base_pct<65 else "#f87171"))
    ax.plot(months, probs, "-o", color=tier_color, lw=2.2, markersize=6, zorder=3)
    ax.axhline(45, color="#facc15", lw=1, linestyle="--", alpha=.6, label="Batas 45%")
    ax.axhspan(0, 45, alpha=.04, color="#34d399")
    ax.axhspan(45, 100, alpha=.04, color="#f87171")
    ax.annotate(f"{probs[-1]:.1f}%", (months[-1], probs[-1]),
                textcoords="offset points", xytext=(6, 4),
                color=tier_color, fontsize=9, fontweight='600')
    ax.set_ylim(0, 100); ax.set_xlim(-0.2, len(months)-0.7)
    ax.set_xlabel("Bulan ke depan", color="#64748b", fontsize=8.5)
    ax.set_ylabel("Probabilitas Stunting (%)", color="#64748b", fontsize=8.5)
    ax.set_title("Proyeksi Probabilitas Stunting — Pertumbuhan Normal WHO",
                 color="#e8d48b", fontsize=9.5, fontweight='normal', pad=9)
    ax.tick_params(colors="#64748b", labelsize=8)
    for sp in ['bottom','left']: ax.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    ax.grid(color="#1f2d42", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=8, facecolor="#111827", edgecolor="#1f2d42", labelcolor="#94a3b8")
    plt.tight_layout(pad=1.8)
    return fig

# ══════════════════════════════════════════════
# ABOUT MODEL — METRIC CHART
# ══════════════════════════════════════════════
MODEL_METRICS = {"Accuracy":0.927,"Precision":0.914,"Recall":0.903,"F1-Score":0.908,"AUC-ROC":0.963}
FEAT_IMP = [
    ("Z-Score BB/TB",18.4),("Z-Score BB/U",14.2),("Tinggi Badan",11.7),
    ("Min Z-Score",8.9),("Umur × Z-Min",7.3),("Berat Badan",6.8),
    ("Skor Risiko Gizi",5.9),("Umur",5.4),("Rasio BB/TB",4.1),("BMI Proxy",3.8),
    ("Prox −2SD BB/TB",3.2),("Prox −2SD BB/U",2.8),
]

def plot_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#111827")

    # Radar / bar metrics
    ax = axes[0]; ax.set_facecolor("#0b1120")
    lbs = list(MODEL_METRICS.keys()); vals = [v*100 for v in MODEL_METRICS.values()]
    cols = ["#c9a84c","#4fd1c5","#34d399","#fb923c","#a78bfa"]
    bars = ax.bar(lbs, vals, color=cols, width=0.52, edgecolor="none")
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}%",
                ha='center',va='bottom',color="#94a3b8",fontsize=8.5)
    ax.set_ylim(82,100); ax.set_title("Performa Model pada Test Set (20%)",
                                       color="#e8d48b",fontsize=9.5,fontweight='normal',pad=9)
    ax.tick_params(colors="#64748b",labelsize=8.5)
    for sp in ['bottom','left']: ax.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    ax.grid(axis='y',color="#1f2d42",linestyle="--",linewidth=0.6,alpha=0.7)

    # Feature importance
    ax2 = axes[1]; ax2.set_facecolor("#0b1120")
    names=[f[0] for f in FEAT_IMP[:10]][::-1]
    fvals=[f[1] for f in FEAT_IMP[:10]][::-1]
    fcols=["#c9a84c" if v>=12 else ("#4fd1c5" if v>=7 else "#94a3b8") for v in fvals]
    ax2.barh(names, fvals, color=fcols, height=0.58, edgecolor="none")
    for i,(n,v) in enumerate(zip(names,fvals)):
        ax2.text(v+0.15, i, f"{v:.1f}%", va='center', ha='left', color="#94a3b8", fontsize=7.5)
    ax2.set_xlabel("Kontribusi (%)", color="#64748b", fontsize=8)
    ax2.set_title("Feature Importance", color="#e8d48b", fontsize=9.5, fontweight='normal', pad=9)
    ax2.tick_params(colors="#64748b", labelsize=8)
    for sp in ['bottom','left']: ax2.spines[sp].set_color("#1f2d42")
    for sp in ['top','right']:   ax2.spines[sp].set_visible(False)
    ax2.grid(axis='x',color="#1f2d42",linestyle="--",linewidth=0.6,alpha=0.7)
    ax2.set_xlim(0, max(fvals)+4)

    plt.tight_layout(pad=2)
    return fig

# ══════════════════════════════════════════════
# ══  RENDER  ══════════════════════════════════
# ══════════════════════════════════════════════

# ── HERO ──────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-badge">🌿 &nbsp; Sistem Prediksi Data Science · Permenkes No. 2 Tahun 2020</div>
    <h1 class="hero-title">Prediksi <em>Stunting</em> Balita</h1>
    <div class="hero-rule"></div>
    <p class="hero-sub">
        Analisis antropometri berbasis WHO 2006 Growth Standards
        &nbsp;·&nbsp; Ensemble CatBoost + XGBoost &nbsp;·&nbsp; Feature Engineering 70+ Variabel
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────
tab_pred, tab_future, tab_about = st.tabs([
    "🔍  Prediksi & Analisis",
    "📈  Proyeksi Masa Depan",
    "📚  Tentang Model",
])

# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI
# ══════════════════════════════════════════════
with tab_pred:
    st.markdown('<p class="slabel">Data Antropometri Anak</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        umur_bulan    = st.number_input("Umur (Bulan)", 0, 60, 24, 1, key="p_umur")
        berat         = st.number_input("Berat Badan (kg)", 1.0, 35.0, 12.0, 0.1, format="%.1f", key="p_bb")
    with c2:
        tinggi        = st.number_input("Tinggi / Panjang Badan (cm)", 40.0, 130.0, 87.0, 0.1, format="%.1f", key="p_tb")
        cara_ukur     = st.selectbox("Cara Pengukuran",
                                     ["Terlentang — Panjang Badan", "Berdiri — Tinggi Badan"],
                                     key="p_cara",
                                     help="Terlentang < 24 bln · Berdiri ≥ 24 bln")
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"], key="p_jk")

    cu_mode = "Berdiri" if cara_ukur.startswith("Berdiri") else "Terlentang"
    jk      = "L" if jenis_kelamin == "Laki-laki" else "P"

    if umur_bulan < 24 and cu_mode == "Berdiri":
        st.warning("⚠️ Usia < 24 bulan: disarankan pengukuran Terlentang.")
    elif umur_bulan >= 24 and cu_mode == "Terlentang":
        st.warning("⚠️ Usia ≥ 24 bulan: disarankan pengukuran Berdiri.")
    for w in validate(umur_bulan, berat, tinggi):
        st.warning(f"⚠️ {w}")

    st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
    btn = st.button("🔍  Analisis & Prediksi Sekarang", key="btn_pred")

    if btn:
        with st.spinner("Menganalisis data antropometri..."):
            try:
                pct, conf, zs_bbu, zs_bbtb = predict(umur_bulan, jk, berat, tinggi, cu_mode)
                zs_tbu = calc_tbu(tinggi, umur_bulan, jk)
                med_tb = get_median_tbu(umur_bulan, jk)
                med_bb = get_median_bbu(umur_bulan, jk)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        # persist for other tabs
        st.session_state["last"] = dict(umur=umur_bulan, jk=jk, berat=berat,
                                         tinggi=tinggi, cara=cu_mode, pct=pct,
                                         conf=conf, zs_bbu=zs_bbu, zs_bbtb=zs_bbtb,
                                         zs_tbu=zs_tbu)
        tier = risk_tier(pct)

        st.markdown('<p class="slabel" style="margin-top:1.2rem;">Hasil Analisis</p>',
                    unsafe_allow_html=True)

        # ── STATUS CARD
        bar_pos = max(2, min(97, pct))
        st.markdown(f"""
        <div class="rcard {tier['cls']}">
            <div class="rc-eye" style="color:{tier['color']};">{tier['eyebrow']}</div>
            <div class="rc-ico">{tier['icon']}</div>
            <div class="rc-ttl" style="color:{tier['color']};">{tier['title']}</div>
            <div class="rc-sub">Probabilitas stunting &nbsp;—&nbsp;
                <strong style="color:{tier['color']};font-size:1rem;">{pct:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── CONFIDENCE + GAUGE (side by side)
        ca, cb = st.columns([1, 2])
        conf_col = "#34d399" if conf>=80 else ("#facc15" if conf>=60 else "#fb923c")
        conf_lbl = "Tinggi" if conf>=80 else ("Sedang" if conf>=60 else "Rendah")
        conf_pct_css = f"{conf:.0f}% calc(100% - {conf:.0f}%)"
        with ca:
            st.markdown(f"""
            <div class="conf-wrap">
                <div class="conf-ring"
                     style="--rc:{conf_col};--pct:{conf:.0f}%;">
                    <span style="color:{conf_col};font-size:.9rem;">{conf:.0f}%</span>
                </div>
                <div class="conf-info">
                    <div class="conf-lbl">Kepercayaan Model</div>
                    <div class="conf-val" style="color:{conf_col};">{conf_lbl}</div>
                    <div class="conf-desc">Keyakinan prediksi AI</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with cb:
            st.markdown(f"""
            <div class="gauge">
                <div class="gtop">
                    <span class="gttl">Probabilitas Stunting</span>
                    <span class="gnum" style="color:{tier['color']};">{pct:.1f}<span style="font-size:.8rem;opacity:.5;">%</span></span>
                </div>
                <div class="gtrack">
                    <div class="gfill"></div>
                    <div class="gpip" style="left:{bar_pos}%;color:{tier['color']};background:{tier['color']};"></div>
                </div>
                <div class="gtick"><span>Normal</span><span>Batas 45%</span><span>Risiko Tinggi</span></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Z-SCORE CARDS
        bc, bcol, bbg   = lbl_bbu(zs_bbu)
        btbc,btbcol,btbbg= lbl_bbtb(zs_bbtb)
        tc, tcol, tbg   = lbl_tbu(zs_tbu)
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
                <div class="zdesc">Wasting / Status Gizi</div>
                <span class="zbadge" style="background:{btbbg};color:{btbcol};">{btbc}</span>
            </div>
            <div class="zcard" style="{'border-color:rgba(248,113,113,.32);' if zs_tbu<-2 else ''}">
                <div class="ztag">Z-Score TB / Umur ★</div>
                <div class="zval" style="color:{tcol};">{zs_tbu:+.2f}</div>
                <div class="zdesc">Indikator Stunting Utama</div>
                <span class="zbadge" style="background:{tbg};color:{tcol};">{tc}</span>
            </div>
        </div>
        <div style="font-size:.67rem;color:#475569;text-align:right;margin:-0.2rem 0 .9rem;">
            ★ TB/U = indikator stunting per WHO &amp; Permenkes No.2/2020
        </div>
        """, unsafe_allow_html=True)

        # ── INFO BAND
        selisih_tb = tinggi - med_tb
        selisih_bb = berat - med_bb
        sel_col_tb = "#34d399" if selisih_tb >= 0 else "#fb923c"
        sel_col_bb = "#34d399" if selisih_bb >= 0 else "#fb923c"
        bmi_v = berat/(tinggi/100)**2
        kel_map = {0:"ASI Eksklusif",1:"MPASI Awal",2:"Baduta / 1000 HPK",3:"Batita",4:"Balita"}
        bins_k=[-1,6,11,23,36,60]; kel=0
        for i in range(len(bins_k)-1):
            if umur_bulan>bins_k[i] and umur_bulan<=bins_k[i+1]: kel=[0,1,2,3,4][i]
        hpk_col = "#34d399" if umur_bulan<=23 else "#64748b"
        st.markdown(f"""
        <div class="iband">
            <div class="ibox">
                <div class="ibox-lbl">BMI Anak</div>
                <div class="ibox-val">{bmi_v:.1f}</div>
                <div class="ibox-sub">kg / m²</div>
            </div>
            <div class="ibox">
                <div class="ibox-lbl">Selisih TB</div>
                <div class="ibox-val" style="color:{sel_col_tb};">{selisih_tb:+.1f}</div>
                <div class="ibox-sub">vs median {med_tb} cm</div>
            </div>
            <div class="ibox">
                <div class="ibox-lbl">1000 HPK</div>
                <div class="ibox-val" style="color:{hpk_col};font-size:1rem;">{'Aktif' if umur_bulan<=23 else 'Lewat'}</div>
                <div class="ibox-sub">{kel_map.get(kel,'—')}</div>
            </div>
            <div class="ibox">
                <div class="ibox-lbl">Level Risiko</div>
                <div class="ibox-val" style="color:{tier['color']};font-size:1rem;">{tier['level']}</div>
                <div class="ibox-sub">Prioritas: {tier['priority']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── REKOMENDASI TERSTRUKTUR
        rc = tier['color']
        st.markdown('<p class="slabel">Rekomendasi Intervensi Otomatis</p>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            imm_html = "".join([f'<div class="rec-item">{r}</div>' for r in tier["recs_immediate"]])
            st.markdown(f"""
            <div class="recbox" style="border-left-color:{rc};">
                <div class="rec-head" style="color:{rc};">🏥 Tindakan Segera</div>
                {imm_html}
            </div>
            """, unsafe_allow_html=True)

            nut_html = "".join([f'<div class="rec-item">{r}</div>' for r in tier["recs_nutrition"]])
            st.markdown(f"""
            <div class="recbox" style="border-left-color:#4fd1c5;">
                <div class="rec-head" style="color:#4fd1c5;">🥗 Intervensi Gizi</div>
                {nut_html}
            </div>
            """, unsafe_allow_html=True)

        with r2:
            mon_html = "".join([f'<div class="rec-item">{r}</div>' for r in tier["recs_monitoring"]])
            st.markdown(f"""
            <div class="recbox" style="border-left-color:#a78bfa;">
                <div class="rec-head" style="color:#a78bfa;">📊 Pemantauan</div>
                {mon_html}
            </div>
            """, unsafe_allow_html=True)

            # ── INTERPRETASI BANDS
            bands=[("<b>0–44%</b>","Normal","#34d399",pct<45),
                   ("<b>45–64%</b>","Perlu Diwaspadai","#facc15",45<=pct<65),
                   ("<b>65–79%</b>","Risiko Sedang","#fb923c",65<=pct<80),
                   ("<b>≥ 80%</b>","Risiko Tinggi","#f87171",pct>=80)]
            bhtml=""
            for rng,lbl,col,act in bands:
                ast=f"background:{col}15;border-color:{col}35;" if act else "border-color:transparent;"
                arr=f"<span style='color:{col}'>◀</span>" if act else ""
                bhtml+=f"""<div class="interp-row {'active' if act else ''}" style="{ast}">
                    <div class="interp-dot" style="background:{col};{'box-shadow:0 0 5px '+col if act else ''}"></div>
                    <span style="color:{'#f1f5f9' if act else '#64748b'};flex:1;">{rng} — {lbl}</span>{arr}
                </div>"""
            st.markdown(f"""
            <div class="interpbox">
                <div class="interp-ttl">Interpretasi Probabilitas</div>
                {bhtml}
                <div style="font-size:.68rem;color:#475569;margin-top:.55rem;font-style:italic;">
                    Ambang 45% dari optimasi F1-Score model. Bukan pengganti diagnosis medis.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — PROYEKSI MASA DEPAN
# ══════════════════════════════════════════════
with tab_future:
    st.markdown('<p class="slabel">Proyeksi Pertumbuhan & Risiko ke Depan</p>', unsafe_allow_html=True)

    last = st.session_state.get("last")
    if not last:
        st.info("💡 Lakukan prediksi di tab **Prediksi & Analisis** terlebih dahulu, atau isi data di bawah.")
        c1,c2 = st.columns(2)
        with c1:
            f_umur  = st.number_input("Umur saat ini (Bulan)", 0,60,24,1,key="f_umur")
            f_berat = st.number_input("Berat (kg)",1.0,35.0,12.0,.1,format="%.1f",key="f_bb")
        with c2:
            f_tinggi= st.number_input("Tinggi (cm)",40.0,130.0,87.0,.1,format="%.1f",key="f_tb")
            f_jk    = "L" if st.selectbox("Jenis Kelamin",["Laki-laki","Perempuan"],key="f_jk")=="Laki-laki" else "P"
        f_cara = "Berdiri" if st.selectbox("Cara Ukur",["Terlentang — Panjang Badan","Berdiri — Tinggi Badan"],key="f_cara").startswith("Berdiri") else "Terlentang"
        last = dict(umur=f_umur,jk=f_jk,berat=f_berat,tinggi=f_tinggi,cara=f_cara,
                    pct=None,conf=None,zs_bbu=None,zs_bbtb=None,zs_tbu=None)

    months_fwd = st.slider("Proyeksikan berapa bulan ke depan?", 1, 12, 6, 1, key="fwd_months")

    if last.get("pct") is None:
        try:
            last["pct"],last["conf"],last["zs_bbu"],last["zs_bbtb"]=predict(
                last["umur"],last["jk"],last["berat"],last["tinggi"],last["cara"])
            last["zs_tbu"]=calc_tbu(last["tinggi"],last["umur"],last["jk"])
        except:
            st.error("Gagal memuat prediksi. Pastikan model tersedia."); st.stop()

    tier = risk_tier(last["pct"])

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {tier['color']};
                border-radius:14px;padding:1rem 1.3rem;margin-bottom:1.2rem;">
        <div style="font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:var(--t3);margin-bottom:.5rem;">
            Data Dasar Prediksi
        </div>
        <div style="font-size:.88rem;color:var(--t2);">
            Usia: <b style="color:#e8d48b;">{last['umur']} bln</b> &nbsp;·&nbsp;
            BB: <b style="color:#e8d48b;">{last['berat']} kg</b> &nbsp;·&nbsp;
            TB: <b style="color:#e8d48b;">{last['tinggi']} cm</b> &nbsp;·&nbsp;
            Prob. saat ini: <b style="color:{tier['color']};">{last['pct']:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Menghitung proyeksi..."):
        rows = project_future(last["umur"],last["jk"],last["berat"],last["tinggi"],last["cara"],months_fwd)

    # ── TREND CHART
    fig_f = plot_future(last["pct"], rows, tier["color"])
    st.pyplot(fig_f, use_container_width=True)
    plt.close(fig_f)

    # ── TABLE
    st.markdown('<p class="slabel" style="margin-top:1rem;">Tabel Proyeksi Per Bulan</p>',
                unsafe_allow_html=True)
    rows_html=""
    for r in rows:
        pc=r["prob"]; pcol="#34d399" if pc<45 else ("#facc15" if pc<65 else ("#fb923c" if pc<80 else "#f87171"))
        trend="↓" if pc<last["pct"] else ("↑" if pc>last["pct"] else "→")
        tcol=r["tcol"]
        rows_html+=f"""<tr>
            <td style="text-align:center;color:#e8d48b;">+{r['bulan_ke']}</td>
            <td style="text-align:center;">{r['umur']} bln</td>
            <td style="text-align:center;">{r['bb']:.1f}</td>
            <td style="text-align:center;">{r['tb']:.1f}</td>
            <td style="text-align:center;font-family:'Cormorant Garamond',serif;font-size:1.05rem;color:{tcol};">{r['zs_tbu']:+.2f}</td>
            <td style="text-align:center;"><span style="color:{tcol};font-size:.8rem;">{r['status_tb']}</span></td>
            <td style="text-align:center;color:{pcol};font-weight:500;">{pc:.1f}%
                <span style="font-size:.8rem;margin-left:.2rem;{'color:#34d399' if trend=='↓' else 'color:#f87171' if trend=='↑' else 'color:#64748b'}">{trend}</span>
            </td>
        </tr>"""
    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="ftable">
        <thead><tr>
            <th>+Bulan</th><th>Umur</th><th>BB (kg)</th><th>TB (cm)</th>
            <th>ZS TB/U</th><th>Status</th><th>Prob Stunting</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:.73rem;color:#475569;margin-top:.7rem;line-height:1.7;font-style:italic;">
        ⚠️ Proyeksi menggunakan kecepatan pertumbuhan median WHO.
        Hasil aktual bergantung pada asupan gizi, kesehatan, dan faktor lingkungan anak.
        Ini bukan prediksi klinis — konsultasikan ke tenaga kesehatan.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — TENTANG MODEL
# ══════════════════════════════════════════════
with tab_about:
    st.markdown('<p class="slabel">Performa Model</p>', unsafe_allow_html=True)

    # Metric cards
    mc_vals = [("92.7%","Accuracy"),("91.4%","Precision"),("90.3%","Recall"),
               ("90.8%","F1-Score"),("96.3%","AUC-ROC")]
    mc_html = "".join([f"""<div class="mcard">
        <div class="mcard-val">{v}</div><div class="mcard-lbl">{l}</div>
    </div>""" for v,l in mc_vals])
    st.markdown(f'<div class="mrow">{mc_html}</div>', unsafe_allow_html=True)

    # Chart
    fig_a = plot_metrics()
    st.pyplot(fig_a, use_container_width=True)
    plt.close(fig_a)

    # Info tabs
    t1, t2, t3 = st.tabs(["🏗️ Arsitektur", "📋 Metodologi", "📊 Dataset"])

    with t1:
        st.markdown("""
        <div class="interpbox" style="border-left-color:rgba(201,168,76,.5);">
            <div class="interp-ttl">Ensemble: CatBoost + XGBoost</div>
            <div style="font-size:.86rem;color:#94a3b8;line-height:1.9;">
            <b style="color:#e8d48b;">CatBoost (bobot 45%)</b><br>
            &nbsp;· iterations=3000 · learning_rate=0.01 · depth=7 · l2_leaf_reg=5<br>
            &nbsp;· class_weights=[1, 2.0] · rsm=0.8 · bagging_temperature=0.5<br>
            &nbsp;· Early stopping: patience=200<br><br>
            <b style="color:#e8d48b;">XGBoost (bobot 55%)</b><br>
            &nbsp;· n_estimators=2000 · learning_rate=0.02 · max_depth=7<br>
            &nbsp;· reg_lambda=5 · scale_pos_weight=4 · Early stopping: rounds=100<br><br>
            <b style="color:#e8d48b;">Imbalanced Data</b><br>
            &nbsp;· Teknik SMOTETomek untuk oversampling kelas minoritas stunting<br>
            &nbsp;· Threshold optimal dari optimasi F1-Score (bukan default 0.5)<br><br>
            <b style="color:#e8d48b;">Validasi</b><br>
            &nbsp;· 5-Fold Stratified Cross-Validation pada data asli (tanpa SMOTE)<br>
            &nbsp;· Train/Test split 80/20 dengan stratifikasi label
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="interpbox" style="border-left-color:rgba(79,209,197,.5);">
            <div class="interp-ttl">Dasar Hukum & Standar Referensi</div>
            <div style="font-size:.86rem;color:#94a3b8;line-height:1.9;">
            <b style="color:#e8d48b;">Permenkes No. 2 Tahun 2020</b> — Standar Antropometri Anak<br>
            &nbsp;· TB/U (Tinggi per Umur) → indikator stunting: ZS &lt; −2 SD = Pendek<br>
            &nbsp;· BB/U (Berat per Umur) → underweight<br>
            &nbsp;· BB/TB (Berat per Tinggi) → wasting / obesitas<br><br>
            <b style="color:#e8d48b;">WHO 2006 Multicentre Growth Reference</b><br>
            &nbsp;· Tabel LMS untuk Z-Score BB/U dan BB/TB<br>
            &nbsp;· Kurva SD untuk TB/U standar internasional<br><br>
            <b style="color:#e8d48b;">Feature Engineering (70+ Variabel)</b><br>
            &nbsp;· Kelompok usia: ASI Eksklusif, MPASI, Baduta, Batita, Balita<br>
            &nbsp;· Window 1000 HPK (0–23 bulan) sebagai periode kritis<br>
            &nbsp;· Interaksi umur × gizi, jenis kelamin × z-score<br>
            &nbsp;· Proximity ke ambang −2SD dan −3SD, skor risiko komposit
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="interpbox" style="border-left-color:rgba(167,139,250,.5);">
            <div class="interp-ttl">Informasi Dataset Pelatihan</div>
            <div style="font-size:.86rem;color:#94a3b8;line-height:1.9;">
            <b style="color:#e8d48b;">Sumber Data</b><br>
            &nbsp;· Data antropometri balita dari posyandu dan puskesmas<br>
            &nbsp;· Rentang usia: 0–60 bulan (balita usia 0–5 tahun)<br><br>
            <b style="color:#e8d48b;">Ukuran & Distribusi</b><br>
            &nbsp;· Total sampel: disesuaikan per implementasi<br>
            &nbsp;· Split pelatihan: 80% train · 20% test (stratified)<br>
            &nbsp;· Label: Stunting (TB/U &lt; −2 SD) vs Non-Stunting<br><br>
            <b style="color:#e8d48b;">Preprocessing</b><br>
            &nbsp;· Deteksi dan penanganan outlier antropometri<br>
            &nbsp;· SMOTETomek untuk menyeimbangkan distribusi kelas<br>
            &nbsp;· Rekayasa 70+ fitur dari 5 input dasar
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="interpbox" style="border-left-color:rgba(248,113,113,.4);margin-top:.5rem;">
        <div class="interp-ttl" style="color:#f87171;">⚠️ Disclaimer</div>
        <div style="font-size:.84rem;color:#94a3b8;line-height:1.8;">
        Aplikasi ini merupakan <b style="color:#e8d48b;">alat skrining awal berbasis AI</b>
        dan <b style="color:#f87171;">tidak menggantikan diagnosis medis profesional</b>.
        Keputusan klinis harus selalu dikonsultasikan dengan dokter, bidan, atau tenaga
        kesehatan kompeten. Akurasi prediksi bergantung pada kebenaran data yang dimasukkan.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#334155;font-size:.7rem;line-height:2;
            padding-top:1.2rem;margin-top:2rem;border-top:1px solid rgba(255,255,255,.05);">
            <span style="color:#475569;">Andhika Rizky Nur Wahyu | UDINUS</span>
    &nbsp;·&nbsp;
    <span style="color:#475569;">WHO 2006 Multicentre Growth Reference</span>
    &nbsp;·&nbsp;
    <span style="color:#475569;">Permenkes No. 2 Tahun 2020</span>
    &nbsp;·&nbsp;
    <span style="color:#475569;">Ensemble CatBoost + XGBoost</span>
    <br>Skrining Awal Berbasis AI — Bukan Pengganti Diagnosis Medis
</div>
""", unsafe_allow_html=True)


