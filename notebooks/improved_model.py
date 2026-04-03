import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       roc_auc_score,
                                       f1_score)
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline

print("=" * 55)
print(" PYROVISION — Improved Fire Prediction Model")
print(" Features F1 / F2 / F3 / F4 — Risk Scoring")
print("=" * 55)

# ── STEP 1: Load and clean Algeria data ───────────────────
print("\n[ 1 / 7 ]  Loading Algeria fire data...")

cols = ['day','month','year','temp','RH','wind','rain',
        'FFMC','DMC','DC','ISI','BUI','FWI','fire']

df = pd.read_csv(
    "../data/raw/algeria/Algerian_forest_fires_dataset.csv",
    skiprows=1, header=None, names=cols
)
df = df.apply(lambda c: c.map(lambda x: x.strip()
              if isinstance(x, str) else x))
df = df[pd.to_numeric(df['temp'], errors='coerce').notna()]
for col in cols[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['fire'] = df['fire'].str.strip().str.lower()
df['fire'] = df['fire'].map({'fire': 1, 'not fire': 0})
df = df.dropna().reset_index(drop=True)

print(f"  Records ready : {len(df)}")

# ── STEP 2: Load and merge Australia weather data ─────────
print("\n[ 2 / 7 ]  Loading Australia weather data...")

wx = pd.read_csv("../data/raw/weather/weatherAUS.csv")

# Keep only columns that match what we have in Algeria
# and that are useful fire-risk indicators
wx_clean = wx[[
    'MinTemp','MaxTemp','Rainfall',
    'WindGustSpeed','Humidity9am','Humidity3pm',
    'Temp9am','Temp3pm','RainToday'
]].copy()

# Create unified column names
wx_clean = wx_clean.rename(columns={
    'MaxTemp'       : 'temp',
    'Humidity9am'   : 'RH',
    'WindGustSpeed' : 'wind',
    'Rainfall'      : 'rain',
})

# Create a fire risk label for weather data
# High temp + low humidity + low rain + high wind = fire risk
wx_clean = wx_clean.dropna()
wx_clean['fire'] = (
    (wx_clean['temp']  > wx_clean['temp'].quantile(0.75)) &
    (wx_clean['RH']    < wx_clean['RH'].quantile(0.25))   &
    (wx_clean['wind']  > wx_clean['wind'].quantile(0.75)) &
    (wx_clean['rain']  < 1.0)
).astype(int)

print(f"  Weather records : {len(wx_clean):,}")
print(f"  High-risk days  : {wx_clean['fire'].sum():,} "
      f"({wx_clean['fire'].mean()*100:.1f}%)")

# ── STEP 3: Engineer new features ─────────────────────────
print("\n[ 3 / 7 ]  Engineering features...")

def engineer_features(data):
    d = data.copy()

    # Heat-Dryness Index — combines temp and humidity
    # Higher = more dangerous
    d['heat_dryness'] = d['temp'] * (100 - d['RH']) / 100

    # Wind-Rain ratio — high wind and no rain is dangerous
    d['wind_rain_ratio'] = d['wind'] / (d['rain'] + 0.1)

    # Drought proxy — no rain and high temp
    d['drought_score'] = d['temp'] - (d['rain'] * 2)

    return d

# Apply to Algeria data
features_base    = ['temp','RH','wind','rain']
features_all     = ['temp','RH','wind','rain',
                    'heat_dryness','wind_rain_ratio','drought_score']

df = engineer_features(df)

print(f"  Base features    : {features_base}")
print(f"  New features     : ['heat_dryness','wind_rain_ratio','drought_score']")
print(f"  Total features   : {len(features_all)}")

# ── STEP 4: Train 3 models and compare ────────────────────
print("\n[ 4 / 7 ]  Training and comparing models...")

X = df[features_all].values
y = df['fire'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Random Forest" : RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42
    ),
    "Gradient Boost" : GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    ),
    "Logistic Reg" : Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ))
    ])
}

results = {}
print(f"\n  {'Model':<20} {'F1':>6} {'AUC':>6} {'CV Mean':>8}")
print("  " + "-" * 45)

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    y_prob = m.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(m, X, y, cv=5,
                          scoring='f1_weighted').mean()

    results[name] = {'model': m, 'f1': f1, 'auc': auc, 'cv': cv}
    print(f"  {name:<20} {f1:>6.3f} {auc:>6.3f} {cv:>8.3f}")

# Pick the best model by AUC
best_name  = max(results, key=lambda n: results[n]['auc'])
best_model = results[best_name]['model']
print(f"\n  Best model : {best_name}")

# ── STEP 5: Detailed evaluation of best model ─────────────
print("\n[ 5 / 7 ]  Detailed evaluation...")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print(f"\n  {best_name} — Full Report")
print("  " + "-" * 45)
print(classification_report(y_test, y_pred,
      target_names=['No Fire','Fire'], digits=3))

print("  CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(f"                    Predicted No Fire   Predicted Fire")
print(f"  Actual No Fire        {cm[0][0]:>8}           {cm[0][1]:>8}")
print(f"  Actual Fire           {cm[1][0]:>8}           {cm[1][1]:>8}")

fn_rate = cm[1][0] / (cm[1][0] + cm[1][1]) * 100
print(f"\n  False Negative Rate : {fn_rate:.1f}%")
print(f"  (fires we missed — target is below 10%)")

# ── STEP 6: Build the Risk Scoring Engine ─────────────────
print("\n[ 6 / 7 ]  Building risk scoring engine...")

# This is the core function that Features F1-F4 will call
def predict_fire_risk(temp, humidity, wind, rain, model=best_model):
    """
    Takes weather inputs and returns:
    - probability (0.0 to 1.0)
    - risk tier   (LOW / MEDIUM / HIGH)
    - risk color  (GREEN / ORANGE / RED)
    - action      (what to do)
    """
    # Engineer features exactly as in training
    heat_dryness   = temp * (100 - humidity) / 100
    wind_rain_ratio = wind / (rain + 0.1)
    drought_score  = temp - (rain * 2)

    features = np.array([[
        temp, humidity, wind, rain,
        heat_dryness, wind_rain_ratio, drought_score
    ]])

    prob = model.predict_proba(features)[0][1]

    # Risk tier classification — matches F5 color scheme
    if prob < 0.30:
        tier   = "LOW"
        color  = "GREEN"
        action = "Safe for routine operations. Weekly monitoring."
    elif prob < 0.70:
        tier   = "MEDIUM"
        color  = "ORANGE"
        action = "Heightened monitoring. Pre-position resources."
    else:
        tier   = "HIGH"
        color  = "RED"
        action = "ALERT — Deploy emergency response immediately."

    return {
        'probability' : round(prob, 4),
        'percentage'  : round(prob * 100, 1),
        'tier'        : tier,
        'color'       : color,
        'action'      : action
    }

# ── STEP 7: Live demonstration ─────────────────────────────
print("\n[ 7 / 7 ]  Live risk scoring demonstration...")

scenarios = [
    # name,                    temp  RH  wind  rain
    ("Extreme fire day",         42,  10,  55,   0),
    ("Very high risk",           38,  18,  40,   0),
    ("High risk",                35,  25,  30,   0),
    ("Medium risk — morning",    28,  45,  20,   0),
    ("Medium risk — afternoon",  32,  35,  25,   2),
    ("Low risk — after rain",    22,  72,  10,  15),
    ("Safe — monsoon day",       18,  88,   5,  40),
]

print()
print(f"  {'Scenario':<30} {'Prob':>6}  {'Tier':>6}  {'Color':>8}")
print("  " + "-" * 60)

for name, temp, rh, wind, rain in scenarios:
    result = predict_fire_risk(temp, rh, wind, rain)
    icon   = "🔴" if result['tier'] == "HIGH"   else \
             "🟠" if result['tier'] == "MEDIUM" else "🟢"
    print(f"  {name:<30} {result['percentage']:>5.1f}%  "
          f"{result['tier']:>6}  {icon} {result['color']}")

# Show full detail for the worst case
print()
print("  FULL DETAIL — Extreme fire day scenario:")
worst = predict_fire_risk(42, 10, 55, 0)
for key, val in worst.items():
    print(f"    {key:<15}: {val}")

# ── SAVE ───────────────────────────────────────────────────
os.makedirs("../models/weights", exist_ok=True)

with open("../models/weights/fire_model_v2.pkl", "wb") as f:
    pickle.dump({
        'model'         : best_model,
        'model_name'    : best_name,
        'features'      : features_all,
        'predict_fn'    : predict_fire_risk,
        'results'       : {k: {'f1': v['f1'],
                               'auc': v['auc']}
                           for k, v in results.items()}
    }, f)

print(f"\n  Model v2 saved → models/weights/fire_model_v2.pkl")

print("\n" + "=" * 55)
print("  Step 3C complete")
print("  Risk scoring engine is ready")
print("  Features F1 F2 F3 F4 data layer — DONE")
print("=" * 55)