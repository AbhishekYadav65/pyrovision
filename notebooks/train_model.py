import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print(" PYROVISION — First Fire Prediction Model")
print("=" * 55)

# ── STEP 1: Load Algeria data with correct headers ─────────
print("\n[ 1 / 6 ]  Loading Algeria fire dataset...")

cols = ['day','month','year','temp','RH','wind','rain',
        'FFMC','DMC','DC','ISI','BUI','FWI','fire']

df = pd.read_csv(
    "../data/raw/algeria/Algerian_forest_fires_dataset.csv",
    skiprows=1,
    header=None,
    names=cols
)

print(f"  Raw shape: {df.shape}")

# ── STEP 2: Clean the data ─────────────────────────────────
print("\n[ 2 / 6 ]  Cleaning data...")

# strip whitespace from all values
df = df.apply(lambda col: col.map(lambda x: x.strip()
              if isinstance(x, str) else x))

# drop rows that are region separator lines
# (they contain text like 'Bejaia' or 'Sidi-Bel')
df = df[pd.to_numeric(df['temp'], errors='coerce').notna()]

# convert all feature columns to numbers
for col in cols[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# clean the target column — make it 0 or 1
df['fire'] = df['fire'].str.strip().str.lower()
df['fire'] = df['fire'].map({'fire': 1, 'not fire': 0})

# drop any remaining nulls
df = df.dropna()
df = df.reset_index(drop=True)

print(f"  Clean shape : {df.shape}")
print(f"  Fire cases  : {df['fire'].sum()} out of {len(df)}")
print(f"  No-fire     : {(df['fire']==0).sum()} out of {len(df)}")
print(f"\n  Sample:")
print(df[['temp','RH','wind','rain','FWI','fire']].head(8).to_string())

# ── STEP 3: Build features ─────────────────────────────────
print("\n[ 3 / 6 ]  Building feature set...")

# These are the weather + fire index features
# Same features we will use from real weather data later
features = ['temp','RH','wind','rain','FFMC','DMC','DC','ISI','BUI','FWI']

X = df[features].values
y = df['fire'].values

print(f"  Features used : {features}")
print(f"  X shape       : {X.shape}")
print(f"  y shape       : {y.shape}")
print(f"  Fire rate     : {y.mean()*100:.1f}%")

# ── STEP 4: Split into train and test ──────────────────────
print("\n[ 4 / 6 ]  Splitting train / test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")

# ── STEP 5: Train the model ────────────────────────────────
print("\n[ 5 / 6 ]  Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # handles fire/no-fire imbalance
    random_state=42
)

model.fit(X_train, y_train)
print("  Training complete")

# ── STEP 6: Evaluate ───────────────────────────────────────
print("\n[ 6 / 6 ]  Evaluating on test set...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n  CLASSIFICATION REPORT")
print("  " + "-" * 45)
print(classification_report(y_test, y_pred,
      target_names=['No Fire','Fire'],
      digits=3))

print("  CONFUSION MATRIX")
print("  " + "-" * 45)
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted No Fire  Predicted Fire")
print(f"  Actual No Fire      {cm[0][0]:>6}            {cm[0][1]:>6}")
print(f"  Actual Fire         {cm[1][0]:>6}            {cm[1][1]:>6}")

# Feature importance
print("\n  FEATURE IMPORTANCE (what matters most)")
print("  " + "-" * 45)
importances = model.feature_importances_
for feat, imp in sorted(zip(features, importances),
                        key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:>6}  {bar}  {imp:.3f}")

# ── SAVE MODEL ─────────────────────────────────────────────
import pickle, os
os.makedirs("../models/weights", exist_ok=True)

with open("../models/weights/fire_model_v1.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n  Model saved → models/weights/fire_model_v1.pkl")

# ── QUICK PREDICTION TEST ───────────────────────────────────
print("\n" + "=" * 55)
print("  LIVE PREDICTION TEST")
print("=" * 55)

# Simulate a dangerous fire day
dangerous = pd.DataFrame([{
    'temp': 38, 'RH': 15, 'wind': 25, 'rain': 0,
    'FFMC': 95, 'DMC': 180, 'DC': 700,
    'ISI': 18, 'BUI': 200, 'FWI': 45
}])

# Simulate a safe day
safe = pd.DataFrame([{
    'temp': 18, 'RH': 75, 'wind': 8, 'rain': 12,
    'FFMC': 40, 'DMC': 20, 'DC': 100,
    'ISI': 2, 'BUI': 25, 'FWI': 5
}])

for label, row in [("Dangerous day (hot, dry, windy)", dangerous),
                   ("Safe day (cool, humid, rainy)",   safe)]:
    prob = model.predict_proba(row[features])[0][1]
    risk = "🔴 HIGH RISK" if prob > 0.7 else \
           "🟠 MEDIUM"    if prob > 0.3 else "🟢 LOW RISK"
    print(f"\n  Scenario : {label}")
    print(f"  Fire probability : {prob*100:.1f}%")
    print(f"  Risk level       : {risk}")

print("\n" + "=" * 55)
print("  Step 3A complete — first model is working")
print("=" * 55)