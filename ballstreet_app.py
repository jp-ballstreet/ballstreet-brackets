"""
ballstreet_app.py — Ball Street Brackets v4
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from collections import Counter

st.set_page_config(page_title="Ball Street Brackets", page_icon="🏀", layout="wide")

OUTPUT_DIR = Path("outputs")
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "b6ac60f1af8f858f067770fcd3aa0333")

TID = {
    "Duke":1181,"Siena":1373,"Ohio State":1326,"TCU":1395,"St. John's":1385,
    "Northern Iowa":1320,"Kansas":1242,"Cal Baptist":1465,"Louisville":1257,
    "South Florida":1378,"Michigan State":1277,"North Dakota St":1295,"UCLA":1417,
    "UCF":1416,"UConn":1163,"Furman":1202,"Arizona":1112,"LIU":1254,
    "Villanova":1437,"Utah State":1429,"Wisconsin":1458,"High Point":1219,
    "Arkansas":1116,"Hawaii":1218,"BYU":1140,"Texas":1400,"NC State":1301,
    "Gonzaga":1211,"Kennesaw State":1244,"Miami":1274,"Missouri":1281,
    "Purdue":1345,"Queens":1474,"Florida":1196,"Lehigh":1250,
    "Prairie View A&M":1341,"Clemson":1155,"Iowa":1234,"Vanderbilt":1435,
    "McNeese":1270,"Nebraska":1304,"Troy":1407,"North Carolina":1314,
    "VCU":1433,"Illinois":1228,"Penn":1335,"Saint Mary's":1388,
    "Texas A&M":1401,"Houston":1222,"Idaho":1225,"Michigan":1276,
    "Howard":1224,"UMBC":1420,"Georgia":1208,"Saint Louis":1387,
    "Texas Tech":1403,"Akron":1103,"Alabama":1104,"Hofstra":1220,
    "Tennessee":1397,"Miami (OH)":1275,"SMU":1374,"Virginia":1438,
    "Wright State":1460,"Kentucky":1246,"Santa Clara":1365,"Iowa State":1235,
    "Tennessee State":1398,
}
ID_TO_NAME = {v: k for k, v in TID.items()}

REGIONS = {
    "East":[("Duke",1),("Siena",16),("Ohio State",8),("TCU",9),
            ("St. John's",5),("Northern Iowa",12),("Kansas",4),("Cal Baptist",13),
            ("Louisville",6),("South Florida",11),("Michigan State",3),("North Dakota St",14),
            ("UCLA",7),("UCF",10),("UConn",2),("Furman",15)],
    "West":[("Arizona",1),("LIU",16),("Villanova",8),("Utah State",9),
            ("Wisconsin",5),("High Point",12),("Arkansas",4),("Hawaii",13),
            ("BYU",6),("NC State",11),("Gonzaga",3),("Kennesaw State",14),
            ("Miami",7),("Missouri",10),("Purdue",2),("Queens",15)],
    "South":[("Florida",1),("Prairie View A&M",16),("Clemson",8),("Iowa",9),
             ("Vanderbilt",5),("McNeese",12),("Nebraska",4),("Troy",13),
             ("North Carolina",6),("VCU",11),("Illinois",3),("Penn",14),
             ("Saint Mary's",7),("Texas A&M",10),("Houston",2),("Idaho",15)],
    "Midwest":[("Michigan",1),("Howard",16),("Georgia",8),("Saint Louis",9),
               ("Texas Tech",5),("Akron",12),("Alabama",4),("Hofstra",13),
               ("Tennessee",6),("SMU",11),("Virginia",3),("Wright State",14),
               ("Kentucky",7),("Santa Clara",10),("Iowa State",2),("Tennessee State",15)],
}
FF_PAIRS = [("East","West"),("South","Midwest")]
RNAMES = {1:"Round of 64",2:"Round of 32",3:"Sweet 16",4:"Elite 8",5:"Final Four",6:"Championship"}


# ═══════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════
@st.cache_data
def load_app_data():
    d = {}
    for key, fn in [("preds","predictions_2026_readable.csv"),("profiles","team_profiles_2026.csv")]:
        p = OUTPUT_DIR / fn
        if p.exists(): d[key] = pd.read_csv(p)
    return d

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_vegas_odds():
    """Fetch NCAAB odds ONCE per day. Returns {team_name: implied_probability}."""
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds",
            params={"regions":"us","markets":"h2h","oddsFormat":"american","apiKey":ODDS_API_KEY},
            timeout=15
        )
        if resp.status_code != 200: return {}
        odds = {}
        for g in resp.json():
            for bk in g.get("bookmakers",[])[:1]:
                for mkt in bk.get("markets",[]):
                    if mkt["key"] == "h2h":
                        for o in mkt["outcomes"]:
                            am = o["price"]
                            prob = 100/(am+100) if am > 0 else abs(am)/(abs(am)+100)
                            odds[o["name"]] = round(prob * 100, 1)
        return odds
    except Exception:
        return {}

def wp(data, a, b):
    preds = data.get("preds", pd.DataFrame())
    if len(preds)==0: return 0.5
    ia, ib = TID.get(a), TID.get(b)
    if not ia or not ib: return 0.5
    lo, hi = min(ia,ib), max(ia,ib)
    m = preds[(preds["TeamA_ID"]==lo)&(preds["TeamB_ID"]==hi)]
    if len(m)==0: return 0.5
    p = m.iloc[0]["TeamA_Pred"]
    return p if ia==lo else 1-p


# ═══════════════════════════════════════════════
# TOURNAMENT PROBABILITIES (pre-computed from 10,000 Monte Carlo simulations)
# ═══════════════════════════════════════════════
@st.cache_data
def load_monte_carlo():
    """Load pre-computed tournament probabilities from 10,000 Monte Carlo simulations.
    Run 08_monte_carlo.py to regenerate after model changes."""
    import json
    mc_path = OUTPUT_DIR / "monte_carlo_results.json"
    if mc_path.exists():
        with open(mc_path) as f:
            mc = json.load(f)
        champ = mc.get("championship", {})
        ff = mc.get("final_four", {})
        n = mc.get("n_sims", 10000)
        return champ, ff, n
    # Fallback: empty
    return {}, {}, 0


# ═══════════════════════════════════════════════
# DETERMINISTIC BRACKET SIMULATION
# ═══════════════════════════════════════════════
@st.cache_data
def simulate(_n):
    """Load pre-built smart bracket. Falls back to deterministic if not available."""
    import json
    smart_path = OUTPUT_DIR / "smart_brackets.json"
    if smart_path.exists():
        with open(smart_path) as f:
            all_brackets = json.load(f)
        return all_brackets
    # Fallback: build deterministic bracket
    data = load_app_data()
    B_chalk = _build_deterministic(data)
    return {"chalk": B_chalk, "balanced": B_chalk, "aggressive": B_chalk}

def _build_deterministic(data):
    B = {}
    for reg, teams in REGIONS.items():
        rnds = {}
        r1 = []
        for i in range(0,16,2):
            na,sa = teams[i]; nb,sb = teams[i+1]
            pa = wp(data, na, nb)
            r1.append(dict(a=na,sa=sa,b=nb,sb=sb,pa=pa,reason="model",
                           w=na if pa>=0.5 else nb, ws=sa if pa>=0.5 else sb))
        rnds["R1"] = r1
        for rn in [2,3,4]:
            prev = rnds[f"R{rn-1}"]
            cur = []
            for i in range(0,len(prev),2):
                na,sa = prev[i]["w"],prev[i]["ws"]
                nb,sb = prev[i+1]["w"],prev[i+1]["ws"]
                pa = wp(data, na, nb)
                cur.append(dict(a=na,sa=sa,b=nb,sb=sb,pa=pa,reason="model",
                                w=na if pa>=0.5 else nb, ws=sa if pa>=0.5 else sb))
            rnds[f"R{rn}"] = cur
        B[reg] = rnds
    ff = []
    for ra, rb in FF_PAIRS:
        ca, cb = B[ra]["R4"][0], B[rb]["R4"][0]
        na,sa,nb,sb = ca["w"],ca["ws"],cb["w"],cb["ws"]
        pa = wp(data, na, nb)
        ff.append(dict(a=na,sa=sa,b=nb,sb=sb,pa=pa,ra=ra,rb=rb,reason="model",
                       w=na if pa>=0.5 else nb, ws=sa if pa>=0.5 else sb))
    B["FF"] = ff
    na,sa = ff[0]["w"],ff[0]["ws"]
    nb,sb = ff[1]["w"],ff[1]["ws"]
    pa = wp(data, na, nb)
    B["CHAMP"] = dict(a=na,sa=sa,b=nb,sb=sb,pa=pa,reason="model",
                       w=na if pa>=0.5 else nb, ws=sa if pa>=0.5 else sb)
    return B


# ═══════════════════════════════════════════════
# GRADES & INSIGHTS
# ═══════════════════════════════════════════════
def get_grades(data, name):
    profiles = data.get("profiles", pd.DataFrame())
    if len(profiles)==0: return None
    tid = TID.get(name)
    if not tid: return None
    m = profiles[profiles["TeamID"]==tid]
    if len(m)==0: return None
    p = m.iloc[0]
    def pct(col, higher=True):
        if col not in profiles.columns: return None
        v = profiles[col]
        if v.std()<0.01: return None
        r = (v<p[col]).sum() if higher else (v>p[col]).sum()
        return int(r/len(v)*100)
    def gr(s):
        if s is None: return None
        for t,l in [(90,"A+"),(80,"A"),(70,"B+"),(60,"B"),(50,"C+"),(40,"C"),(30,"D+"),(20,"D")]:
            if s>=t: return l
        return "F"
    specs = [("Overall","KP_AdjEM",True),("Offense","KP_AdjOE",True),
             ("Defense","KP_AdjDE",False),("Shooting","EFG_pct",True),
             ("3-Point Shooting","THREE_PT_pct",True),("Perimeter Defense","THREE_PT_pctD",False),
             ("Rebounding","OREB_pct",True),("Ball Security","TOV_pct",False),
             ("Coaching","coach_PAKE",True),("Win Record","WIN_pct",True)]
    cats = {}
    for label, col, higher in specs:
        s = pct(col, higher)
        g = gr(s)
        if s is not None: cats[label] = dict(score=s, grade=g)
    return dict(name=name, seed=int(p["SEED"]), overall=pct("KP_AdjEM",True) or 50,
                adj_em=round(p.get("KP_AdjEM",0),1), cats=cats)

def build_insights(data, a, b):
    ga, gb = get_grades(data, a), get_grades(data, b)
    pa = wp(data, a, b)
    if not ga or not gb: return []
    fav = a if pa>=0.5 else b
    dog = b if fav==a else a
    fg = ga if fav==a else gb
    dg = gb if fav==a else ga
    ins = []
    gap = abs(fg.get("overall",50)-dg.get("overall",50))
    fo = fg["cats"].get("Overall",{}).get("grade","C")
    do = dg["cats"].get("Overall",{}).get("grade","C")
    if gap>40: ins.append(f"**{fav}** ({fo} overall) is far stronger than {dog} ({do}). Expect a comfortable win.")
    elif gap>25: ins.append(f"**{fav}** ({fo}) has a clear talent edge over {dog} ({do}), but March is unpredictable.")
    elif gap>12: ins.append(f"Competitive game. {fav} ({fo}) is better on paper, but {dog} ({do}) can hang.")
    else: ins.append(f"Genuine toss-up. {fav} ({fo}) vs {dog} ({do}) — could go either way.")
    shared = set(fg["cats"])&set(dg["cats"])-{"Overall"}
    edges = [(c, fg["cats"][c]["score"]-dg["cats"][c]["score"],
              fg["cats"][c]["grade"], dg["cats"][c]["grade"]) for c in shared]
    edges.sort(key=lambda x:x[1], reverse=True)
    if edges and edges[0][1]>15:
        c,_,g1,g2 = edges[0]
        ins.append(f"**{fav}'s biggest edge — {c}:** {g1} vs {dog}'s {g2}.")
    edges.sort(key=lambda x:x[1])
    if edges and edges[0][1]<-15:
        c,_,g1,g2 = edges[0]
        ins.append(f"**{dog}'s path to the upset — {c}:** {g2} vs {fav}'s {g1}.")
    dog_3 = dg["cats"].get("3-Point Shooting",{})
    fav_pd = fg["cats"].get("Perimeter Defense",{})
    if dog_3.get("score",0)>65 and fav_pd.get("score",0)<50:
        ins.append(f"🎯 {dog} shoots well from three ({dog_3['grade']}) and {fav}'s perimeter D is weak ({fav_pd['grade']}).")
    elif dog_3.get("score",0)>70:
        ins.append(f"🎯 {dog} is a {dog_3['grade']} three-point team — one hot night could flip this.")
    fc = fg["cats"].get("Coaching",{}); dc = dg["cats"].get("Coaching",{})
    if fc.get("score",50)>75 and dc.get("score",50)<40:
        ins.append(f"🎓 Coaching edge: {fav}'s coach ({fc['grade']}) has a strong March track record vs {dog}'s ({dc['grade']}).")
    return ins[:5]


# ═══════════════════════════════════════════════
# BRACKET HTML (fixed alignment with CSS grid)
# ═══════════════════════════════════════════════
def bracket_html(B, region):
    """Proper bracket with CSS grid for alignment."""
    rounds = [B[region].get(f"R{r}",[]) for r in range(1,5)]
    labels = ["ROUND OF 64","ROUND OF 32","SWEET 16","ELITE 8"]

    # Each game is a fixed-height cell. Later rounds have taller spacing to align.
    game_h = 58  # px per game card
    gap = 6

    css = f"""<style>
    .bkt{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;overflow-x:auto;padding:4px;gap:0}}
    .bkt-col{{display:flex;flex-direction:column;min-width:195px;padding:0 6px}}
    .bkt-lbl{{font-size:9px;font-weight:700;letter-spacing:1.5px;color:#6b7280;text-align:center;
              padding:4px 0;text-transform:uppercase}}
    .bkt-g{{background:#111827;border:1px solid #1e293b;border-radius:5px;overflow:hidden;flex-shrink:0}}
    .bkt-t{{display:flex;align-items:center;padding:4px 8px;font-size:12px;gap:4px;
            border-bottom:1px solid #1e293b}}
    .bkt-t:last-child{{border-bottom:none}}
    .bkt-t.w{{background:#0c1629;color:#e2e8f0;font-weight:700}}
    .bkt-t.l{{color:#4b5563}}
    .bkt-s{{color:#6b7280;font-size:10px;min-width:16px;text-align:right}}
    .bkt-n{{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
    .bkt-p{{font-size:10px;font-weight:600;min-width:28px;text-align:right}}
    .bkt-t.w .bkt-p{{color:#34d399}}
    .bkt-t.l .bkt-p{{color:#4b5563}}
    .bkt-conn{{display:flex;flex-direction:column;width:18px;min-width:18px}}
    .bkt-line{{border-right:2px solid #374151;border-top:2px solid #374151;
               border-bottom:2px solid #374151;border-radius:0 4px 4px 0}}
    .bkt-champ{{display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-width:120px;padding:8px}}
    .bkt-champ-icon{{font-size:28px}}
    .bkt-champ-name{{font-size:13px;font-weight:800;color:#e2e8f0;margin-top:2px}}
    .bkt-champ-sub{{font-size:10px;color:#6b7280}}
    </style>"""

    def game_card(g):
        pa = float(g.get("pa", 0.5))
        ac = "w" if g["w"]==g["a"] else "l"
        bc = "w" if g["w"]==g["b"] else "l"
        is_upset = g.get("reason","") in ("upset_pick","strategic_upset","model_upset")
        border = "border-left:3px solid #f59e0b;" if is_upset else ""
        fire = ' 🔥' if is_upset else ''
        return f'''<div class="bkt-g" style="{border}">
            <div class="bkt-t {ac}"><span class="bkt-s">{g["sa"]}</span><span class="bkt-n">{g["a"]}</span><span class="bkt-p">{pa*100:.0f}%</span></div>
            <div class="bkt-t {bc}"><span class="bkt-s">{g["sb"]}</span><span class="bkt-n">{g["b"]}{fire}</span><span class="bkt-p">{(1-pa)*100:.0f}%</span></div>
        </div>'''

    SLOT = game_h + gap  # 64px per R1 game slot
    LH = 22  # label div height (must match .bkt-lbl padding + font)

    html = css + '<div class="bkt">'

    for ri, (games, label) in enumerate(zip(rounds, labels)):
        # Connector lines between rounds
        if ri > 0:
            conn_h = (2 ** (ri-1)) * SLOT - 4  # subtract 4px for border-top(2) + border-bottom(2)
            # First connector top = label_h + center of first game in previous round
            first_mt = LH + game_h//2 + ((2**(ri-1)) - 1) * SLOT // 2
            between_mt = (2**(ri-1)) * SLOT  # gap between consecutive arms

            html += '<div class="bkt-conn">'
            for i in range(len(games)):
                mt = first_mt if i == 0 else between_mt
                html += f'<div class="bkt-line" style="height:{conn_h}px;margin-top:{mt}px"></div>'
            html += '</div>'

        # Game column - center each game between its two feeder games
        first_pad = ((2 ** ri) - 1) * SLOT // 2
        game_spacing = (2 ** ri) * SLOT - game_h

        html += f'<div class="bkt-col"><div class="bkt-lbl">{label}</div>'
        for gi, g in enumerate(games):
            mt = first_pad if gi == 0 else game_spacing
            html += f'<div style="margin-top:{mt}px">{game_card(g)}</div>'
        html += '</div>'

    # Champion
    if B[region].get("R4"):
        ch = B[region]["R4"][0]
        ch_pa = float(ch.get("pa", 0.5))
        wp_ch = ch_pa*100 if ch["w"]==ch["a"] else (1-ch_pa)*100
        champ_mt = LH + game_h//2 + 3*SLOT + SLOT//2
        html += f'''<div class="bkt-conn"><div class="bkt-line" style="height:30px;margin-top:{champ_mt}px"></div></div>
        <div class="bkt-champ" style="margin-top:{champ_mt - 15}px">
            <div class="bkt-champ-icon">🏆</div>
            <div class="bkt-champ-name">({ch["ws"]}) {ch["w"]}</div>
            <div class="bkt-champ-sub">{wp_ch:.0f}% in Elite 8</div>
        </div>'''

    html += '</div>'
    return html


def ff_html(B):
    ff = B.get("FF",[]); ch = B.get("CHAMP",{})
    html = '''<style>
    .ff-wrap{font-family:-apple-system,sans-serif;text-align:center;padding:12px}
    .ff-row{display:flex;justify-content:center;gap:32px;margin-bottom:20px}
    .ff-gm{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:12px 16px;min-width:210px;text-align:left}
    .ff-lbl{font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}
    .ff-t{display:flex;justify-content:space-between;padding:5px 0;font-size:13px}
    .ff-t.w{color:#e2e8f0;font-weight:700}.ff-t.l{color:#4b5563}
    .ff-t .p{color:#34d399}.ff-t.l .p{color:#4b5563}
    .ch-box{display:inline-block;background:linear-gradient(135deg,#111827,#1e1b4b);
            border:2px solid #34d399;border-radius:12px;padding:18px 36px;margin-top:8px}
    .ch-tag{font-size:10px;color:#34d399;text-transform:uppercase;letter-spacing:2px}
    .ch-name{font-size:24px;font-weight:900;color:#e2e8f0;margin:4px 0}
    .ch-sub{color:#6b7280;font-size:12px}
    </style><div class="ff-wrap"><h3 style="color:#e2e8f0;margin-bottom:12px">🏀 Final Four</h3><div class="ff-row">'''
    for g in ff:
        pa = float(g.get("pa", 0.5))
        ac = "w" if g["w"]==g["a"] else "l"
        bc = "w" if g["w"]==g["b"] else "l"
        html += f'''<div class="ff-gm"><div class="ff-lbl">{g.get("ra","")} vs {g.get("rb","")}</div>
            <div class="ff-t {ac}"><span>({g["sa"]}) {g["a"]}</span><span class="p">{pa*100:.0f}%</span></div>
            <div class="ff-t {bc}"><span>({g["sb"]}) {g["b"]}</span><span class="p">{(1-pa)*100:.0f}%</span></div></div>'''
    html += '</div>'
    if ch:
        ch_pa = float(ch.get("pa", 0.5))
        html += f'''<div class="ch-box"><div class="ch-tag">🏆 Predicted Champion</div>
            <div class="ch-name">({ch["ws"]}) {ch["w"]}</div>
            <div class="ch-sub">{max(ch_pa,1-ch_pa)*100:.0f}% in the title game</div></div>'''
    html += '</div>'
    return html


# ═══════════════════════════════════════════════
# MATCHUP DETAIL
# ═══════════════════════════════════════════════
def show_matchup(data, a, sa, b, sb, vegas=None):
    pa = wp(data, a, b); pb = 1-pa
    fav = a if pa>=0.5 else b
    fpct = max(pa,pb)*100
    if fpct>=85: cl,ce = "Strong Favorite","🟢"
    elif fpct>=65: cl,ce = "Likely Winner","🔵"
    elif fpct>=55: cl,ce = "Slight Edge","🟡"
    else: cl,ce = "Toss-Up","🔴"

    st.markdown(f"### {ce} Ball Street Pick: **{fav}** ({fpct:.0f}%) — {cl}")

    # Vegas comparison
    if vegas:
        va, vb = vegas.get(a), vegas.get(b)
        if va and vb:
            total = va + vb
            va_pct, vb_pct = va/total*100, vb/total*100
            diff = pa*100 - va_pct
            edge_team = a if diff > 0 else b
            edge_amt = abs(diff)
            edge_str = f"🎰 **Vegas:** {a} {va_pct:.0f}% / {b} {vb_pct:.0f}%"
            if edge_amt > 3:
                edge_str += f" · **Ball Street edge: {edge_amt:.0f}% more on {edge_team}**"
            st.markdown(edge_str)

    c1,c2,c3 = st.columns([2,5,2])
    with c1: st.markdown(f"**({sa}) {a}**")
    with c2: st.progress(pa)
    with c3: st.markdown(f"<div style='text-align:right'><b>({sb}) {b}</b></div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([2,5,2])
    with c1: st.markdown(f"**{pa*100:.0f}%**")
    with c3: st.markdown(f"<div style='text-align:right'><b>{pb*100:.0f}%</b></div>", unsafe_allow_html=True)

    ga, gb = get_grades(data, a), get_grades(data, b)
    if ga and gb:
        shared = [c for c in ga["cats"] if c in gb["cats"]]
        rows = ""
        for cat in shared:
            sca,scb = ga["cats"][cat]["score"],gb["cats"][cat]["score"]
            gra,grb = ga["cats"][cat]["grade"],gb["cats"][cat]["grade"]
            ab = "font-weight:800" if sca>scb+5 else ""
            bb = "font-weight:800" if scb>sca+5 else ""
            ck_a = " ✓" if sca>scb+5 else ""
            ck_b = " ✓" if scb>sca+5 else ""
            bar_a = f'<div style="background:#34d399;height:4px;width:{sca}%;border-radius:2px;margin-top:2px"></div>'
            bar_b = f'<div style="background:#60a5fa;height:4px;width:{scb}%;border-radius:2px;margin-top:2px"></div>'
            rows += f'<tr><td style="padding:5px 8px;{ab};font-size:13px">{gra}{ck_a}{bar_a}</td><td style="padding:5px 8px;text-align:center;font-size:11px;color:#9ca3af">{cat}</td><td style="padding:5px 8px;text-align:right;{bb};font-size:13px">{grb}{ck_b}{bar_b}</td></tr>'
        st.html(f'<table style="width:100%;border-collapse:collapse;font-family:-apple-system,sans-serif"><thead><tr><th style="padding:5px 8px;text-align:left;color:#34d399;font-size:12px">({sa}) {a}</th><th style="text-align:center;color:#6b7280;font-size:10px">CATEGORY</th><th style="text-align:right;color:#60a5fa;font-size:12px;padding:5px 8px">({sb}) {b}</th></tr></thead><tbody>{rows}</tbody></table>')

    insights = build_insights(data, a, b)
    if insights:
        st.markdown("#### The Breakdown")
        for i in insights: st.markdown(f"- {i}")


# ═══════════════════════════════════════════════
# TAB: BALL STREET BRACKET
# ═══════════════════════════════════════════════
def tab_bracket(data, all_brackets, vegas):
    st.markdown("## 🏀 Ball Street Bracket")
    st.markdown("*Our model's predicted path — with strategic upsets built in*")

    mode = st.radio("Bracket Style",["chalk","balanced","aggressive"],
                    format_func={"chalk":"📋 Chalk (safest)","balanced":"🎯 Balanced (recommended)","aggressive":"🔥 Aggressive (max upsets)"}.get,
                    horizontal=True, label_visibility="collapsed")

    B = all_brackets.get(mode, all_brackets.get("balanced", {}))
    if not B:
        st.error("No bracket data found. Run 11_smart_bracket.py first.")
        return

    # Quick summary bar
    ch = B.get("CHAMP", {})
    ff_teams = [g["w"] for g in B.get("FF", [])]
    n_upsets = sum(1 for r in ["East","West","South","Midwest"]
                   for rnd in ["R1","R2","R3","R4"]
                   for g in B.get(r,{}).get(rnd,[])
                   if g.get("reason","") in ("upset_pick","strategic_upset","model_upset"))
    if ch:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Predicted Champion", f"({ch['ws']}) {ch['w']}")
        with cols[1]:
            st.metric("Final Four", " · ".join(ff_teams) if ff_teams else "—")
        with cols[2]:
            st.metric("Upset Picks", f"{n_upsets} games")
        st.markdown("")

    view = st.radio("",["Regional Brackets","Final Four & Championship"],horizontal=True,label_visibility="collapsed")
    if view == "Final Four & Championship":
        st.html(ff_html(B))
        st.markdown("---")
        for g in B.get("FF",[]):
            with st.expander(f"🏀 {g['ra']} vs {g['rb']}: ({g['sa']}) {g['a']} vs ({g['sb']}) {g['b']}"):
                show_matchup(data, g["a"], g["sa"], g["b"], g["sb"], vegas)
        ch = B.get("CHAMP",{})
        if ch:
            with st.expander(f"🏆 Championship: ({ch['sa']}) {ch['a']} vs ({ch['sb']}) {ch['b']}"):
                show_matchup(data, ch["a"], ch["sa"], ch["b"], ch["sb"], vegas)
    else:
        region = st.selectbox("Region",["East","West","South","Midwest"],key="bs_reg")
        st.html(bracket_html(B, region))
        st.markdown("---")
        rnd = st.selectbox("Explore matchups",list(range(1,5)),format_func=lambda x:RNAMES[x],key="bs_rnd")
        for g in B[region].get(f"R{rnd}",[]):
            is_upset = g.get("reason","") in ("upset_pick","strategic_upset","model_upset")
            label = f"{'🔥 ' if is_upset else ''}({g['sa']}) {g['a']} vs ({g['sb']}) {g['b']}"
            with st.expander(label):
                show_matchup(data, g["a"], g["sa"], g["b"], g["sb"], vegas)


# ═══════════════════════════════════════════════
# TAB: MAKE YOUR OWN
# ═══════════════════════════════════════════════
def tab_make_own(data, vegas):
    st.markdown("## ✏️ Make My Own Bracket")
    st.markdown("*Pick winners round by round. Ball Street probabilities shown to guide you.*")

    # Initialize all region picks
    for reg in REGIONS:
        sk = f"myo_{reg}"
        if sk not in st.session_state: st.session_state[sk] = {}
    if "myo_ff" not in st.session_state: st.session_state["myo_ff"] = {}

    region = st.selectbox("Region",["East","West","South","Midwest"],key="mo_reg")
    teams = REGIONS[region]
    picks = st.session_state[f"myo_{region}"]

    def do_round(rnd, matchups, prefix=""):
        st.markdown(f"### {RNAMES[rnd]}")
        winners = []
        for gi, (na, sa, nb, sb) in enumerate(matchups):
            if na is None or nb is None:
                winners.append((None, None))
                continue
            pa = wp(data, na, nb)
            pk = f"{prefix}R{rnd}_{gi}"
            c1,c2,c3 = st.columns([5,2,5])
            with c1:
                is_p = picks.get(pk)==na
                if st.button(f"{'✅ ' if is_p else ''}({sa}) {na}", key=f"mo_{region}_{pk}_a", use_container_width=True):
                    picks[pk] = na
                    for r in range(rnd+1, 7):
                        for k in list(picks.keys()):
                            if k.startswith(f"{prefix}R{r}_"): del picks[k]
                    st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;padding-top:6px;color:#6b7280;font-size:11px'>{pa*100:.0f}–{(1-pa)*100:.0f}</div>", unsafe_allow_html=True)
            with c3:
                is_p = picks.get(pk)==nb
                if st.button(f"{'✅ ' if is_p else ''}({sb}) {nb}", key=f"mo_{region}_{pk}_b", use_container_width=True):
                    picks[pk] = nb
                    for r in range(rnd+1, 7):
                        for k in list(picks.keys()):
                            if k.startswith(f"{prefix}R{r}_"): del picks[k]
                    st.rerun()
            with st.expander("Compare"):
                show_matchup(data, na, sa, nb, sb, vegas)
            w = picks.get(pk)
            winners.append((w, sa if w==na else sb if w==nb else None))
        return winners

    # Regional rounds
    r1m = [(teams[i*2][0],teams[i*2][1],teams[i*2+1][0],teams[i*2+1][1]) for i in range(8)]
    r1w = do_round(1, r1m)
    prev = r1w
    for rnd in [2,3,4]:
        if any(w is None for w,_ in prev):
            st.info(f"Pick all winners above to unlock {RNAMES[rnd]}.")
            break
        matchups = [(prev[i*2][0],prev[i*2][1],prev[i*2+1][0],prev[i*2+1][1])
                    for i in range(len(prev)//2)]
        prev = do_round(rnd, matchups)

    st.markdown("---")
    total = sum(1 for v in picks.values() if v)
    st.progress(min(total/15,1.0))
    st.caption(f"{total}/15 picks for {region}")
    if picks.get("R4_0"):
        st.success(f"🏆 Your {region} champion: **{picks['R4_0']}**")

    # Final Four + Championship (cross-region)
    st.markdown("---")
    st.markdown("## 🏀 Final Four & Championship")

    region_champs = {}
    for reg in REGIONS:
        rp = st.session_state.get(f"myo_{reg}", {})
        champ = rp.get("R4_0")
        if champ:
            # Find seed
            for n, s in REGIONS[reg]:
                if n == champ:
                    region_champs[reg] = (champ, s)
                    break
            if reg not in region_champs:
                region_champs[reg] = (champ, "?")

    missing = [r for r in ["East","West","South","Midwest"] if r not in region_champs]
    if missing:
        st.info(f"Complete these regions first: {', '.join(missing)}")
        for reg in ["East","West","South","Midwest"]:
            if reg in region_champs:
                st.markdown(f"✅ **{reg}:** {region_champs[reg][0]}")
            else:
                st.markdown(f"⬜ **{reg}:** *(not yet picked)*")
    else:
        ff_picks = st.session_state["myo_ff"]

        # FF Game 1: East vs West
        st.markdown("### Final Four")
        for gi, (ra, rb) in enumerate(FF_PAIRS):
            na, sa = region_champs[ra]
            nb, sb = region_champs[rb]
            pa = wp(data, na, nb)
            pk = f"FF_{gi}"

            st.markdown(f"**{ra} vs {rb}**")
            c1,c2,c3 = st.columns([5,2,5])
            with c1:
                is_p = ff_picks.get(pk)==na
                if st.button(f"{'✅ ' if is_p else ''}({sa}) {na}", key=f"mo_ff_{gi}_a", use_container_width=True):
                    ff_picks[pk] = na
                    ff_picks.pop("CHAMP_0", None)
                    st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;padding-top:6px;color:#6b7280;font-size:11px'>{pa*100:.0f}–{(1-pa)*100:.0f}</div>", unsafe_allow_html=True)
            with c3:
                is_p = ff_picks.get(pk)==nb
                if st.button(f"{'✅ ' if is_p else ''}({sb}) {nb}", key=f"mo_ff_{gi}_b", use_container_width=True):
                    ff_picks[pk] = nb
                    ff_picks.pop("CHAMP_0", None)
                    st.rerun()
            with st.expander("Compare"):
                show_matchup(data, na, sa, nb, sb, vegas)

        # Championship
        ff_w0 = ff_picks.get("FF_0")
        ff_w1 = ff_picks.get("FF_1")
        if ff_w0 and ff_w1:
            st.markdown("### 🏆 Championship")
            # Find seeds
            s0 = s1 = "?"
            for reg in REGIONS:
                for n,s in REGIONS[reg]:
                    if n==ff_w0: s0=s
                    if n==ff_w1: s1=s
            pa = wp(data, ff_w0, ff_w1)
            pk = "CHAMP_0"
            c1,c2,c3 = st.columns([5,2,5])
            with c1:
                is_p = ff_picks.get(pk)==ff_w0
                if st.button(f"{'✅ ' if is_p else ''}({s0}) {ff_w0}", key="mo_champ_a", use_container_width=True):
                    ff_picks[pk] = ff_w0; st.rerun()
            with c2:
                st.markdown(f"<div style='text-align:center;padding-top:6px;color:#6b7280;font-size:11px'>{pa*100:.0f}–{(1-pa)*100:.0f}</div>", unsafe_allow_html=True)
            with c3:
                is_p = ff_picks.get(pk)==ff_w1
                if st.button(f"{'✅ ' if is_p else ''}({s1}) {ff_w1}", key="mo_champ_b", use_container_width=True):
                    ff_picks[pk] = ff_w1; st.rerun()
            with st.expander("Compare"):
                show_matchup(data, ff_w0, s0, ff_w1, s1, vegas)

            if ff_picks.get("CHAMP_0"):
                st.success(f"## 🏆 Your Champion: **{ff_picks['CHAMP_0']}**")
                st.balloons()
        elif ff_w0 or ff_w1:
            st.info("Pick both Final Four winners to unlock the Championship game.")



# ═══════════════════════════════════════════════
# TAB: VIEW MY BRACKET
# ═══════════════════════════════════════════════
def user_bracket_html(region):
    """Render user picks as a bracket visualization."""
    picks = st.session_state.get(f"myo_{region}", {})
    teams = REGIONS[region]
    game_h, gap = 58, 6

    css = """<style>
    .ubkt{font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;overflow-x:auto;padding:4px;gap:0;align-items:flex-start}
    .ubkt-col{display:flex;flex-direction:column;min-width:195px;padding:0 4px}
    .ubkt-lbl{font-size:9px;font-weight:700;letter-spacing:1.5px;color:#6b7280;text-align:center;padding:4px 0;text-transform:uppercase;height:22px}
    .ubkt-g{background:#111827;border:1px solid #1e293b;border-radius:5px;overflow:hidden}
    .ubkt-t{display:flex;align-items:center;padding:4px 8px;font-size:12px;gap:4px;border-bottom:1px solid #1e293b}
    .ubkt-t:last-child{border-bottom:none}
    .ubkt-t.picked{background:#0c1629;color:#34d399;font-weight:700}
    .ubkt-t.not-picked{color:#4b5563}
    .ubkt-t.pending{color:#9ca3af;font-style:italic}
    .ubkt-s{color:#6b7280;font-size:10px;min-width:16px;text-align:right}
    .ubkt-n{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .ubkt-conn{display:flex;flex-direction:column;width:16px;min-width:16px}
    .ubkt-line{border-right:2px solid #374151;border-top:2px solid #374151;border-bottom:2px solid #374151;border-radius:0 3px 3px 0}
    .ubkt-champ{display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:110px;padding:8px}
    </style>"""

    SLOT = game_h + gap
    LH = 22
    labels = ["ROUND OF 64","ROUND OF 32","SWEET 16","ELITE 8"]

    # Build all rounds from picks
    all_rounds = []

    # R1
    r1 = []
    for i in range(8):
        na, sa = teams[i*2]
        nb, sb = teams[i*2+1]
        w = picks.get(f"R1_{i}")
        r1.append(dict(a=na, sa=sa, b=nb, sb=sb, w=w))
    all_rounds.append(r1)

    # R2-R4
    for rnd in range(2, 5):
        prev = all_rounds[-1]
        cur = []
        for i in range(0, len(prev), 2):
            g1, g2 = prev[i], prev[i+1]
            na = g1["w"]
            sa = None
            if na:
                sa = g1["sa"] if na == g1["a"] else g1["sb"] if na == g1["b"] else "?"
            nb = g2["w"]
            sb = None
            if nb:
                sb = g2["sa"] if nb == g2["a"] else g2["sb"] if nb == g2["b"] else "?"
            w = picks.get(f"R{rnd}_{i//2}")
            cur.append(dict(a=na or "TBD", sa=sa or "?", b=nb or "TBD", sb=sb or "?", w=w))
        all_rounds.append(cur)

    def card(g):
        if g["a"] == "TBD" and g["b"] == "TBD":
            return '<div class="ubkt-g"><div class="ubkt-t pending"><span class="ubkt-n">TBD</span></div><div class="ubkt-t pending"><span class="ubkt-n">TBD</span></div></div>'
        ac = "picked" if g["w"] == g["a"] else "not-picked" if g["w"] else "pending"
        bc = "picked" if g["w"] == g["b"] else "not-picked" if g["w"] else "pending"
        a_str = g["a"] if g["a"] != "TBD" else "TBD"
        b_str = g["b"] if g["b"] != "TBD" else "TBD"
        sa_str = g["sa"] if g["sa"] != "?" else ""
        sb_str = g["sb"] if g["sb"] != "?" else ""
        return f'''<div class="ubkt-g">
            <div class="ubkt-t {ac}"><span class="ubkt-s">{sa_str}</span><span class="ubkt-n">{a_str}</span></div>
            <div class="ubkt-t {bc}"><span class="ubkt-s">{sb_str}</span><span class="ubkt-n">{b_str}</span></div>
        </div>'''

    html = css + '<div class="ubkt">'
    for ri, (games, label) in enumerate(zip(all_rounds, labels)):
        if ri > 0:
            conn_h = (2 ** (ri-1)) * SLOT - 4
            first_mt = LH + game_h//2 + ((2**(ri-1)) - 1) * SLOT // 2
            between_mt = (2**(ri-1)) * SLOT
            html += '<div class="ubkt-conn">'
            for i in range(len(games)):
                mt = first_mt if i == 0 else between_mt
                html += f'<div class="ubkt-line" style="height:{conn_h}px;margin-top:{mt}px"></div>'
            html += '</div>'

        first_pad = ((2 ** ri) - 1) * SLOT // 2
        game_spacing = (2 ** ri) * SLOT - game_h
        html += f'<div class="ubkt-col"><div class="ubkt-lbl">{label}</div>'
        for gi, g in enumerate(games):
            mt = first_pad if gi == 0 else game_spacing
            html += f'<div style="margin-top:{mt}px">{card(g)}</div>'
        html += '</div>'

    # Champion
    if len(all_rounds) >= 4 and all_rounds[3] and all_rounds[3][0].get("w"):
        ch = all_rounds[3][0]
        champ_mt = LH + game_h//2 + 3*SLOT + SLOT//2
        html += f'''<div class="ubkt-conn"><div class="ubkt-line" style="height:30px;margin-top:{champ_mt}px"></div></div>
        <div class="ubkt-champ" style="margin-top:{champ_mt - 15}px">
            <div style="font-size:28px">🏆</div>
            <div style="font-size:13px;font-weight:800;color:#34d399">{ch["w"]}</div>
        </div>'''

    html += '</div>'
    return html


def tab_view_bracket(data):
    st.markdown("## 📋 View My Bracket")
    st.markdown("*Your picks visualized*")

    # Count total picks
    total_picks = 0
    max_picks = 63  # 32+16+8+4+2+1
    region_champs = {}
    for reg in REGIONS:
        rp = st.session_state.get(f"myo_{reg}", {})
        total_picks += sum(1 for v in rp.values() if v)
        if rp.get("R4_0"):
            region_champs[reg] = rp["R4_0"]
    ff_picks = st.session_state.get("myo_ff", {})
    total_picks += sum(1 for v in ff_picks.values() if v)

    st.markdown(f"**Progress: {total_picks} picks made**")
    st.progress(min(total_picks / max_picks, 1.0))

    # Region brackets
    for reg in ["East", "West", "South", "Midwest"]:
        rp = st.session_state.get(f"myo_{reg}", {})
        n_picks = sum(1 for v in rp.values() if v)
        champ = rp.get("R4_0", "—")
        status = f"✅ Champion: **{champ}**" if champ != "—" else f"⬜ {n_picks}/15 picks"
        with st.expander(f"**{reg} Region** — {status}", expanded=n_picks > 0):
            if n_picks > 0:
                st.html(user_bracket_html(reg))
            else:
                st.info(f"No picks yet for {reg}. Go to 'Make My Own' to start picking!")

    # Final Four summary
    st.markdown("---")
    st.markdown("### 🏀 Final Four & Championship")
    if len(region_champs) == 4:
        for gi, (ra, rb) in enumerate(FF_PAIRS):
            ca, cb = region_champs.get(ra, "?"), region_champs.get(rb, "?")
            ff_w = ff_picks.get(f"FF_{gi}")
            if ff_w:
                st.markdown(f"**{ra} vs {rb}:** ✅ **{ff_w}** advances")
            else:
                st.markdown(f"**{ra} vs {rb}:** {ca} vs {cb} *(not yet picked)*")

        champ = ff_picks.get("CHAMP_0")
        if champ:
            st.markdown(f"## 🏆 Your Champion: **{champ}**")
        elif ff_picks.get("FF_0") and ff_picks.get("FF_1"):
            st.markdown(f"**Championship:** {ff_picks['FF_0']} vs {ff_picks['FF_1']} *(not yet picked)*")
    else:
        missing = [r for r in ["East","West","South","Midwest"] if r not in region_champs]
        st.info(f"Complete all four regions to see Final Four. Missing: {', '.join(missing)}")


# ═══════════════════════════════════════════════
# TAB: EXPLORER
# ═══════════════════════════════════════════════
def tab_explorer(data, vegas):
    st.markdown("## ⚔️ Head-to-Head Explorer")
    st.markdown("*Pick any two tournament teams and see how they match up*")
    all_t = sorted(TID.keys())
    c1,c2 = st.columns(2)
    with c1: a = st.selectbox("Team A",all_t,index=all_t.index("Duke"),key="exa")
    with c2: b = st.selectbox("Team B",[t for t in all_t if t!=a],key="exb")
    sa = sb = "?"
    for _,tl in REGIONS.items():
        for n,s in tl:
            if n==a: sa=s
            if n==b: sb=s
    show_matchup(data, a, sa, b, sb, vegas)


# ═══════════════════════════════════════════════
# TAB: UPSET WATCH
# ═══════════════════════════════════════════════
def tab_upsets(data, vegas):
    st.markdown("## 🔥 Upset Watch")
    st.markdown("*First-round games where the underdog has a real shot*")
    upsets = []
    for reg, teams in REGIONS.items():
        for i in range(0,16,2):
            na,sa = teams[i]; nb,sb = teams[i+1]
            pa = wp(data, na, nb)
            if sa<sb: fav,fs,dog,ds,dp = na,sa,nb,sb,1-pa
            else: fav,fs,dog,ds,dp = nb,sb,na,sa,pa
            gap = ds-fs
            if dp>0.25 and gap>=3:
                upsets.append(dict(r=reg,dog=dog,ds=ds,fav=fav,fs=fs,dp=dp*100,gap=gap))
    upsets.sort(key=lambda x:x["dp"],reverse=True)
    for u in upsets:
        p = u["dp"]
        if p>=50: em,tag = "🔥🔥","OUR MODEL PICKS THE UPSET"
        elif p>=40: em,tag = "🔥","HIGH ALERT"
        elif p>=33: em,tag = "⚠️","WATCH"
        else: em,tag = "👀","LONGSHOT"
        c1,c2 = st.columns([5,1])
        with c1:
            st.markdown(f"{em} **({u['ds']}) {u['dog']}** over ({u['fs']}) {u['fav']}")
            st.caption(f"{u['r']} Region · {tag}")
        with c2: st.markdown(f"### {p:.0f}%")
        with st.expander("Full breakdown"):
            show_matchup(data, u["fav"], u["fs"], u["dog"], u["ds"], vegas)
        st.markdown("---")


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
def render_sidebar(data, B, champ_pcts, ff_pcts, n_sims, vegas):
    if vegas:
        st.sidebar.success(f"🎰 Vegas odds loaded ({len(vegas)} teams)")
    else:
        st.sidebar.info("Vegas odds not available")

    if n_sims:
        st.sidebar.caption(f"Probabilities from {n_sims:,} tournament simulations")

    # Champion probabilities (from Monte Carlo)
    st.sidebar.markdown("### 🏆 Championship Odds")
    top_champs = sorted(champ_pcts.items(), key=lambda x:x[1], reverse=True)[:8]
    champ_html = '<div style="font-family:-apple-system,sans-serif;font-size:13px">'
    for name, pct in top_champs:
        # MC keys are team names; find seed from bracket
        n = name  # Already a name string
        seed = None
        for _,tl in REGIONS.items():
            for tn,ts in tl:
                if tn==n: seed=ts
        if seed:
            bar_w = min(pct * 3, 100)  # Scale so ~33% fills the bar
            champ_html += f'''<div style="margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                    <span>({seed}) <b>{n}</b></span><span style="color:#34d399">{pct:.0f}%</span>
                </div>
                <div style="background:#1e293b;border-radius:3px;height:6px;overflow:hidden">
                    <div style="background:#34d399;height:100%;width:{bar_w}%;border-radius:3px"></div>
                </div>
            </div>'''
    champ_html += '</div>'
    st.sidebar.html(champ_html)

    st.sidebar.markdown("---")

    # Final Four likelihood
    st.sidebar.markdown("### 🏀 Final Four Favorites")
    top_ff = sorted(ff_pcts.items(), key=lambda x:x[1], reverse=True)[:6]
    for name, pct in top_ff:
        seed = None
        for _,tl in REGIONS.items():
            for tn,ts in tl:
                if tn==name: seed=ts
        if seed:
            st.sidebar.markdown(f"({seed}) {name} — {pct:.0f}%")

    st.sidebar.markdown("---")

    # Cinderellas (seeds 7+ with >1% FF chance)
    st.sidebar.markdown("### 🐴 Cinderella Watch")
    cinderellas = []
    for name, pct in ff_pcts.items():
        for _,tl in REGIONS.items():
            for tn,ts in tl:
                if tn==name and ts >= 7 and pct > 1:
                    cinderellas.append((name, ts, pct))
    cinderellas.sort(key=lambda x:x[2], reverse=True)
    if cinderellas:
        for n, s, p in cinderellas[:5]:
            st.sidebar.markdown(f"({s}) **{n}** — {p:.0f}% to Final Four")
    else:
        st.sidebar.caption("No strong Cinderella candidates")

    st.sidebar.markdown("---")

    # Upset alerts (first round games where underdog > 35%)
    st.sidebar.markdown("### ⚡ Upset Alerts")
    upset_alerts = []
    for reg, teams in REGIONS.items():
        for i in range(0,16,2):
            na,sa = teams[i]; nb,sb = teams[i+1]
            pa = wp(data, na, nb)
            if sa < sb:  # na is higher seed (favorite)
                dog, ds, dp = nb, sb, (1-pa)*100
                gap = sb - sa
            else:
                dog, ds, dp = na, sa, pa*100
                gap = sa - sb
            if dp > 35 and gap >= 3:
                upset_alerts.append((dog, ds, dp, reg))
    upset_alerts.sort(key=lambda x: x[2], reverse=True)
    for dog, ds, dp, reg in upset_alerts[:8]:
        st.sidebar.markdown(f"🔥 ({ds}) **{dog}** — {dp:.0f}%")

    st.sidebar.markdown("---")

    # Vegas edge games
    if vegas:
        st.sidebar.markdown("### 🎰 Ball Street vs Vegas")
        st.sidebar.caption("Games where we disagree with Vegas")
        edge_games = []
        for reg, teams in REGIONS.items():
            for i in range(0,16,2):
                na,sa = teams[i]; nb,sb = teams[i+1]
                pa = wp(data, na, nb)
                va, vb = vegas.get(na), vegas.get(nb)
                if va and vb:
                    total = va+vb
                    vegas_a = va/total*100
                    diff = pa*100 - vegas_a
                    if abs(diff) > 5:
                        edge_team = na if diff>0 else nb
                        edge_games.append((edge_team, abs(diff), reg))
        edge_games.sort(key=lambda x:x[1], reverse=True)
        for team, edge, reg in edge_games[:6]:
            st.sidebar.markdown(f"**{team}** +{edge:.0f}% vs Vegas")


# ═══════════════════════════════════════════════
# TAB: ABOUT
# ═══════════════════════════════════════════════
def tab_about():
    st.markdown("## ℹ️ About Ball Street Brackets")

    st.markdown("""
### How to Use This App

**🏀 Ball Street Bracket** — Our model's predicted bracket for the 2026 tournament.
Toggle between three styles: **Chalk** picks the model favorite every game,
**Balanced** adds ~7 strategic upset picks (matching the historical average),
and **Aggressive** goes all-in on chaos. Expand any matchup to see the full breakdown.

**✏️ Make My Own** — Build your bracket round by round. Select a region and pick
winners by clicking team names. Our win probabilities are shown next to each matchup
to help you decide. Complete all four regions to unlock Final Four and Championship picks.

**📋 My Bracket** — See all your picks visualized in a bracket format, with a summary
of your progress and champion.

**⚔️ H2H Explorer** — Compare any two tournament teams head-to-head. See how they
grade out across offense, defense, shooting, and more.

**🔥 Upset Watch** — Every first-round game ranked by upset potential. We combine
our model probability, historical upset rates for that seed matchup, and
team-specific factors to identify the most likely upsets.

---

### How the Model Works

Our predictions come from a **machine learning ensemble** trained on every
NCAA Tournament game from 2008 to 2025 (excluding 2020, which was canceled).

**The process in a nutshell:**

1. **Data collection** — We pull team ratings from KenPom (the gold standard for
college basketball analytics), Barttorvik, and official NCAA box scores. Each team
gets a profile covering ~50 statistical categories.

2. **Feature engineering** — For every possible matchup, we compute the *difference*
between the two teams across categories like offensive efficiency, defensive efficiency,
shooting accuracy, rebounding, ball security, and more. We also factor in momentum
(last 10 games, win streak, conference tournament performance) and coaching track record.

3. **Model training** — We train three different models — XGBoost (a gradient boosting
algorithm), LightGBM (another boosting approach), and Logistic Regression — then blend
them into an optimized ensemble. The blend is tuned to minimize prediction error using
leave-one-season-out cross-validation, meaning when we predict 2024, the model has
never seen 2024 data.

4. **Calibration** — Raw model outputs are passed through isotonic calibration so that
when we say a team has a 70% chance, they actually win about 70% of the time.

5. **Tournament simulation** — We run 10,000 Monte Carlo simulations of the full bracket,
randomly drawing each game result based on our probabilities. This gives us the true
championship and Final Four odds you see in the sidebar.

---

### Key Numbers

| Metric | Value |
|---|---|
| Training data | 1,129 tournament games (2008–2025) |
| Features per matchup | 51 |
| Cross-validated accuracy | 74% |
| Cross-validated log loss | 0.50 |
| Simulations for championship odds | 10,000 |

---

### What the Grades Mean

When you open a matchup breakdown, each team gets letter grades (A+ through F)
across categories like Offense, Defense, and Shooting. These grades are
**percentile ranks among the 68 tournament teams** — not absolute ratings.

- **A+** = Top 10% of tournament teams in this category
- **A** = Top 20%
- **B+** = Top 30%
- **C** = Average for a tournament team
- **F** = Bottom of the tournament field

A team can be "F" in a category and still be a good team overall — it just means
they're the weakest tournament team in that specific area.

---

### The Upset Framework

Our upset scoring combines three factors:

- **Model probability (45%)** — What our ML model thinks the underdog's win chance is
- **Historical rate (30%)** — How often this seed matchup produces upsets
  (for example, 12-seeds beat 5-seeds 41% of the time historically)
- **Team factors (25%)** — Three-point shooting, defensive strength, low turnovers,
  experienced coach, and whether the favorite has vulnerabilities

Games scoring above 40 are automatic upset picks. Games between 30–40 are "lean"
candidates where we pick a few more to hit the historical average of ~7 first-round upsets.

---

### Vegas Odds

When available, we show Vegas implied probabilities alongside our model's predictions.
The **"Ball Street edge"** tells you where our model disagrees with the betting market
by more than 5 percentage points — these are the games where we see value that the
market might be missing.

Vegas odds are refreshed once every 24 hours to conserve API usage.

---

*Built by Ball Street Sports · March 2026*
""")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
def main():
    st.markdown("# 🏀 Ball Street Brackets")
    st.markdown("**2026 NCAA Tournament** · Powered by data, built for fans")
    st.markdown("---")

    data = load_app_data()
    if not data.get("preds") is not None or len(data.get("preds",[]))==0:
        st.error("No predictions found. Run the pipeline first.")
        return

    all_brackets = simulate(len(data["preds"]))
    B = all_brackets.get("balanced", all_brackets.get("chalk", {}))
    vegas = fetch_vegas_odds()
    champ_pcts, ff_pcts, n_sims = load_monte_carlo()

    render_sidebar(data, B, champ_pcts, ff_pcts, n_sims, vegas)

    t1,t2,t3,t4,t5,t6 = st.tabs(["🏀 Ball Street Bracket","✏️ Make My Own","📋 My Bracket","⚔️ H2H Explorer","🔥 Upset Watch","ℹ️ About"])
    with t1: tab_bracket(data, all_brackets, vegas)
    with t2: tab_make_own(data, vegas)
    with t3: tab_view_bracket(data)
    with t4: tab_explorer(data, vegas)
    with t5: tab_upsets(data, vegas)
    with t6: tab_about()

    st.markdown("---")
    st.caption("Ball Street Sports · ML predictions from 2008-2025 NCAA tournament data · Not gambling advice")

if __name__=="__main__": main()
