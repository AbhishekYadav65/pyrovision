import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print(" PYROVISION — Fire Cause Analysis")
print(" Feature F7 — Cause + Prevention Tips")
print("=" * 55)

# ── STEP 1: Load the wildfire database ────────────────────
print("\n[ 1 / 5 ]  Connecting to wildfire database...")

db_path = "../data/raw/viirs/FPA_FOD_20170508.sqlite"
conn    = sqlite3.connect(db_path)

# Load only the columns we actually need
# Full table has 1.88M rows — we select smart
query = """
SELECT
    FIRE_YEAR,
    DISCOVERY_DOY,
    STAT_CAUSE_DESCR,
    FIRE_SIZE,
    FIRE_SIZE_CLASS,
    LATITUDE,
    LONGITUDE,
    STATE
FROM Fires
WHERE LATITUDE IS NOT NULL
  AND LONGITUDE IS NOT NULL
  AND STAT_CAUSE_DESCR IS NOT NULL
  AND STAT_CAUSE_DESCR != 'Missing/Undefined'
"""

print("  Loading records... (may take 30 seconds)")
df = pd.read_sql(query, conn)
conn.close()

print(f"  Records loaded  : {len(df):,}")
print(f"  Years covered   : {df['FIRE_YEAR'].min()} to {df['FIRE_YEAR'].max()}")
print(f"  States covered  : {df['STATE'].nunique()}")
print(f"  Causes found    : {df['STAT_CAUSE_DESCR'].nunique()}")

# ── STEP 2: Overall cause breakdown ───────────────────────
print("\n[ 2 / 5 ]  Analysing fire causes...")

cause_counts = df['STAT_CAUSE_DESCR'].value_counts()
cause_pct    = (cause_counts / len(df) * 100).round(2)

print("\n  CAUSE BREAKDOWN — All 1.88M fires")
print("  " + "-" * 50)
for cause, count in cause_counts.items():
    pct = cause_pct[cause]
    bar = "█" * int(pct / 2)
    print(f"  {cause:<30} {count:>8,}  {pct:>5.1f}%  {bar}")

# ── STEP 3: Seasonal pattern — which month is most dangerous
print("\n[ 3 / 5 ]  Seasonal fire pattern...")

# Convert day-of-year to month
df['month'] = pd.to_datetime(
    df['FIRE_YEAR'].astype(str) + '-' +
    df['DISCOVERY_DOY'].astype(str),
    format='%Y-%j'
).dt.month

month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

monthly = df.groupby('month').agg(
    fire_count=('FIRE_YEAR','count'),
    avg_size  =('FIRE_SIZE','mean')
).round(2)

print("\n  MONTHLY FIRE PATTERN")
print("  " + "-" * 45)
for month, row in monthly.iterrows():
    bar  = "█" * int(row['fire_count'] / 15000)
    name = month_names[month]
    print(f"  {name}  {bar}  {row['fire_count']:>8,} fires  "
          f"avg size {row['avg_size']:.1f} acres")

# ── STEP 4: Biggest fires by cause ────────────────────────
print("\n[ 4 / 5 ]  Largest fires by cause...")

big_fires = df.groupby('STAT_CAUSE_DESCR').agg(
    total_fires =('FIRE_YEAR',  'count'),
    avg_size    =('FIRE_SIZE',  'mean'),
    max_size    =('FIRE_SIZE',  'max'),
    total_acres =('FIRE_SIZE',  'sum')
).sort_values('total_acres', ascending=False).round(1)

print("\n  CAUSE vs TOTAL ACRES BURNED")
print("  " + "-" * 65)
print(f"  {'Cause':<30} {'Fires':>8} {'Avg Acres':>10} {'Total Acres':>14}")
print("  " + "-" * 65)
for cause, row in big_fires.iterrows():
    print(f"  {cause:<30} {row['total_fires']:>8,} "
          f"{row['avg_size']:>10.1f} "
          f"{row['total_acres']:>14,.0f}")

# ── STEP 5: Prevention tips engine ────────────────────────
print("\n[ 5 / 5 ]  Building prevention tips engine...")

# This is the core of Feature F7
# Maps each cause to specific actionable prevention advice
PREVENTION_TIPS = {
    "Lightning": {
        "risk"   : "Natural ignition — cannot be prevented but can be anticipated",
        "before" : [
            "Monitor IMD lightning forecast alerts 48 hours ahead",
            "Pre-position water tankers at ridge access roads before storm season",
            "Schedule helicopter surveillance within 24 hours after any lightning storm",
            "Install lightning conductor rods at hilltop ranger stations",
        ],
        "during" : [
            "Deploy ground teams to check ridges and peaks within 2 hours of storm",
            "Watch for smoke plumes using VIIRS near-real-time alerts",
        ]
    },

    "Debris Burning": {
        "risk"   : "Controlled burns escaping — most preventable cause",
        "before" : [
            "Enforce burn permits only on days with wind below 15 km/h",
            "Ban debris burning when FWI exceeds 25",
            "Require 100m clearance buffer from forest edge for all burns",
            "Register all planned burns with forest department 48 hours prior",
        ],
        "during" : [
            "Maintain standby water source within 500m of any active burn",
            "Never leave a debris fire unattended",
        ]
    },

    "Arson": {
        "risk"   : "Deliberate ignition — requires community and surveillance response",
        "before" : [
            "Install CCTV cameras at known repeat ignition locations",
            "Increase foot patrols at forest boundaries during high-risk season",
            "Engage village forest protection committees for tip-off networks",
            "Coordinate with local police for deterrence patrols",
        ],
        "during" : [
            "Report suspicious activity immediately to Forest Range Officer",
            "Document GPS coordinates of ignition point for investigation",
        ]
    },

    "Campfire": {
        "risk"   : "Tourist and recreational fire — controllable with regulation",
        "before" : [
            "Close high-risk trail zones to camping during FWI > 30 days",
            "Install fire-safe campfire rings at designated sites only",
            "Place bilingual signage at all forest entry points",
            "Brief tour operators on fire safety protocols before season",
        ],
        "during" : [
            "Enforce complete campfire ban when wind exceeds 20 km/h",
            "Ranger check of all campsites at dawn and dusk",
        ]
    },

    "Equipment Use": {
        "risk"   : "Machinery sparks — highest risk during harvest and construction",
        "before" : [
            "Require spark arrestors on all machinery operating near forest",
            "Schedule heavy equipment work only before 11 AM during dry season",
            "Maintain 50m clearance from dry vegetation for all machinery",
        ],
        "during" : [
            "Stop all machinery immediately if FWI exceeds 35",
            "Keep a water backpack with every machinery operator",
        ]
    },

    "Children": {
        "risk"   : "Accidental ignition — school and community awareness critical",
        "before" : [
            "Run fire safety awareness programs in schools near forest areas",
            "Install visible warning signage at village boundary paths",
            "Engage Anganwadi workers to communicate fire risks to families",
        ],
        "during" : [
            "Restrict unsupervised access to forest edge during peak season",
        ]
    },

    "Smoking": {
        "risk"   : "Discarded cigarettes — simple signage and enforcement reduces this",
        "before" : [
            "Place no-smoking signs every 200m on all forest trails",
            "Fine system for smoking violations in forest zones",
            "Install cigarette disposal bins at all forest entry points",
        ],
        "during" : [
            "Rangers to enforce no-smoking during red-alert days",
        ]
    },

    "Railroad": {
        "risk"   : "Spark from tracks — infrastructure coordination required",
        "before" : [
            "Clear 20m vegetation buffer along all railway lines through forest",
            "Request Railways to schedule track maintenance before dry season",
            "Install fire detection sensors along high-risk rail corridors",
        ],
        "during" : [
            "Alert Railways to reduce speed on forest sections during high FWI days",
        ]
    },

    "Miscellaneous": {
        "risk"   : "Mixed and unknown causes — general preparedness applies",
        "before" : [
            "Conduct pre-season firebreak clearing along all division boundaries",
            "Ensure all water points and tanks are filled before dry season",
            "Test all communication equipment before fire season begins",
        ],
        "during" : [
            "Follow standard incident command protocols",
            "Maintain daily contact with neighbouring range officers",
        ]
    },
}

# Demonstrate the tips engine
print("\n  PREVENTION TIPS ENGINE — DEMO")
print("  " + "=" * 55)

top_3_causes = cause_counts.head(3).index.tolist()

for cause in top_3_causes:
    # match to our tips dictionary (partial match)
    matched_key = next(
        (k for k in PREVENTION_TIPS if k.lower() in cause.lower()
         or cause.lower() in k.lower()),
        "Miscellaneous"
    )
    tips = PREVENTION_TIPS[matched_key]

    print(f"\n  🔥 CAUSE : {cause}")
    print(f"     Risk  : {tips['risk']}")
    print(f"     BEFORE fire season:")
    for tip in tips['before'][:2]:
        print(f"       → {tip}")
    print(f"     DURING high-risk days:")
    for tip in tips['during'][:1]:
        print(f"       → {tip}")
    print()

# ── SAVE RESULTS ───────────────────────────────────────────
os.makedirs("../models/weights", exist_ok=True)

import pickle
cause_data = {
    'cause_counts'    : cause_counts.to_dict(),
    'cause_pct'       : cause_pct.to_dict(),
    'big_fires'       : big_fires.to_dict(),
    'monthly_pattern' : monthly.to_dict(),
    'prevention_tips' : PREVENTION_TIPS,
}

with open("../models/weights/cause_analysis.pkl", "wb") as f:
    pickle.dump(cause_data, f)

print("  Cause analysis saved → models/weights/cause_analysis.pkl")

# ── SAVE CHARTS ────────────────────────────────────────────
os.makedirs("../docs", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("PYROVISION — Fire Cause Analysis", fontsize=16, fontweight='bold')

# Pie chart
colors = ['#E74C3C','#E67E22','#F1C40F','#2ECC71','#3498DB',
          '#9B59B6','#1ABC9C','#E91E63','#FF5722']
axes[0].pie(cause_counts.values[:9],
            labels=cause_counts.index[:9],
            autopct='%1.1f%%',
            colors=colors,
            startangle=140)
axes[0].set_title("Fire Causes — Share of 1.88M incidents")

# Monthly bar chart
months = [month_names[m] for m in monthly.index]
axes[1].bar(months, monthly['fire_count'],
            color=['#E74C3C' if m in ['Jun','Jul','Aug']
                   else '#E67E22' if m in ['Apr','May','Sep']
                   else '#3498DB' for m in months])
axes[1].set_title("Monthly Fire Frequency")
axes[1].set_ylabel("Number of fires")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("../docs/cause_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

print("  Chart saved    → docs/cause_analysis.png")

print("\n" + "=" * 55)
print("  Step 3B complete — cause analysis ready")
print("  Feature F7 data layer is built")
print("=" * 55)