"""
08_monte_carlo.py — Run 10,000 tournament simulations and save results.
Run this after any model changes to update championship probabilities.

Usage: python 08_monte_carlo.py
"""
import pandas as pd
import numpy as np
import json
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("outputs")

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
    "East":[(1181,1),(1373,16),(1326,8),(1395,9),(1385,5),(1320,12),(1242,4),(1465,13),
            (1257,6),(1378,11),(1277,3),(1295,14),(1417,7),(1416,10),(1163,2),(1202,15)],
    "West":[(1112,1),(1254,16),(1437,8),(1429,9),(1458,5),(1219,12),(1116,4),(1218,13),
            (1140,6),(1301,11),(1211,3),(1244,14),(1274,7),(1281,10),(1345,2),(1474,15)],
    "South":[(1196,1),(1341,16),(1155,8),(1234,9),(1435,5),(1270,12),(1304,4),(1407,13),
             (1314,6),(1433,11),(1228,3),(1335,14),(1388,7),(1401,10),(1222,2),(1225,15)],
    "Midwest":[(1276,1),(1224,16),(1208,8),(1387,9),(1403,5),(1103,12),(1104,4),(1220,13),
               (1397,6),(1374,11),(1438,3),(1460,14),(1246,7),(1365,10),(1235,2),(1398,15)],
}
FF_PAIRS = [("East","West"),("South","Midwest")]


def main():
    N = 10000
    print(f"Running {N:,} tournament simulations...")

    preds = pd.read_csv(OUTPUT_DIR / "predictions_2026_readable.csv")

    def wp(a_id, b_id):
        lo, hi = min(a_id, b_id), max(a_id, b_id)
        m = preds[(preds["TeamA_ID"] == lo) & (preds["TeamB_ID"] == hi)]
        if len(m) == 0: return 0.5
        p = m.iloc[0]["TeamA_Pred"]
        return p if a_id == lo else 1 - p

    np.random.seed(42)
    champ_ct, ff_ct = Counter(), Counter()

    for sim in range(N):
        if sim % 2000 == 0:
            print(f"  {sim:,}/{N:,} simulations...")
        region_winners = {}
        for reg, teams in REGIONS.items():
            bracket = []
            for i in range(0, 16, 2):
                a_id, a_s = teams[i]; b_id, b_s = teams[i+1]
                p = wp(a_id, b_id)
                bracket.append((a_id, a_s) if np.random.random() < p else (b_id, b_s))
            for rnd in range(3):
                nxt = []
                for i in range(0, len(bracket), 2):
                    a, b = bracket[i], bracket[i+1]
                    p = wp(a[0], b[0])
                    nxt.append(a if np.random.random() < p else b)
                bracket = nxt
            region_winners[reg] = bracket[0]
            ff_ct[bracket[0][0]] += 1

        ff_w = []
        for ra, rb in FF_PAIRS:
            a, b = region_winners[ra], region_winners[rb]
            p = wp(a[0], b[0])
            ff_w.append(a if np.random.random() < p else b)

        a, b = ff_w[0], ff_w[1]
        p = wp(a[0], b[0])
        champ = a if np.random.random() < p else b
        champ_ct[champ[0]] += 1

    def to_dict(counter):
        return {ID_TO_NAME.get(k, str(k)): round(v / N * 100, 1)
                for k, v in counter.most_common() if v / N * 100 >= 0.1}

    result = {
        "n_sims": N,
        "championship": to_dict(champ_ct),
        "final_four": to_dict(ff_ct),
    }

    out = OUTPUT_DIR / "monte_carlo_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {out}")
    print(f"\nChampionship Odds (top 10):")
    for name, pct in list(result["championship"].items())[:10]:
        print(f"  {name:<20s} {pct}%")
    print(f"\nFinal Four Odds (top 10):")
    for name, pct in list(result["final_four"].items())[:10]:
        print(f"  {name:<20s} {pct}%")


if __name__ == "__main__":
    main()
