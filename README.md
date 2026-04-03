# 🔥 PYROVISION
### Forest Fire Prediction & Simulation System
**AI/ML Hackathon — Problem Statement 3**

---

## What This Project Does

PYROVISION predicts forest fire probability and simulates fire spread using AI/ML.  
It takes weather conditions as input and outputs a risk score (LOW / MEDIUM / HIGH) for any region.

Built for the AI/ML Hackathon Problem Statement 3:
- Generate a next-day fire probability map at 30m resolution
- Simulate fire spread over 1, 2, 3, 6, and 12 hours from high-risk zones

Full feature list and technical document is in `/docs/` folder.

---

## Project Structure

```
pyrovision/
├── data/
│   ├── raw/
│   │   ├── viirs/        ← 1.88M US wildfire records (SQLite)
│   │   ├── algeria/      ← Algeria fire dataset (CSV) — used for model training
│   │   ├── weather/      ← Australia weather dataset (CSV)
│   │   ├── lulc/         ← Land cover image classification dataset
│   │   └── dem/          ← SRTM elevation raster files (.tif)
│   └── processed/
│       ├── features/     ← aligned 30m feature stacks (to be built)
│       ├── labels/       ← VIIRS binary fire label rasters (to be built)
│       └── rasters/      ← final COG prediction outputs (to be built)
│
├── models/
│   └── weights/
│       ├── fire_model_v1.pkl     ← Basic Random Forest (Step 3A)
│       ├── fire_model_v2.pkl     ← Improved model with risk scoring (Step 3C)
│       └── cause_analysis.pkl   ← Cause breakdown + prevention tips (Step 3B)
│
├── notebooks/
│   ├── explore_data.py      ← Data exploration script (run first)
│   ├── train_model.py       ← First model training (Step 3A)
│   ├── cause_analysis.py    ← Fire cause analysis (Step 3B)
│   └── improved_model.py    ← Improved model with risk tiers (Step 3C)
│
├── pipeline/           ← data ingestion scripts (next to build)
├── simulation/         ← Cellular Automata engine (next to build)
├── backend/            ← FastAPI server (next to build)
├── frontend/           ← React dashboard (next to build)
├── docs/
│   └── cause_analysis.png   ← Generated cause breakdown chart
├── .env                ← API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ✅ What Has Been Done So Far

### Step 1 — Repository Setup ✅
- GitHub repo created and cloned
- Full folder structure created
- `.gitignore`, `.env`, `requirements.txt` all configured
- Branching strategy set up: `main` → `dev` → `feature/*`

### Step 2 — Data Collection ✅
All 5 datasets downloaded into correct folders:

| Dataset | Location | What It Is |
|---|---|---|
| 188M US Wildfires | `data/raw/viirs/FPA_FOD_20170508.sqlite` | 1.88M fire records with cause, size, lat/lon |
| Algeria Fire Data | `data/raw/algeria/Algerian_forest_fires_dataset.csv` | Labeled fire/no-fire with weather features |
| Australia Weather | `data/raw/weather/weatherAUS.csv` | 145K daily weather readings |
| Land Cover LULC | `data/raw/lulc/intel-image-classification/` | Image classification dataset |
| SRTM Elevation DEM | `data/raw/dem/srtm_57_07.tif` + `srtm_58_07.tif` | Terrain elevation rasters |

### Step 3A — First Working Model ✅
**File:** `notebooks/train_model.py`  
**Model saved:** `models/weights/fire_model_v1.pkl`

- Cleaned Algeria CSV (fixed broken headers, removed separator rows)
- Trained Random Forest classifier on fire/no-fire labels
- Model correctly predicts HIGH risk for hot/dry/windy days
- Model correctly predicts LOW risk for cool/humid/rainy days

### Step 3B — Fire Cause Analysis ✅
**File:** `notebooks/cause_analysis.py`  
**Output saved:** `models/weights/cause_analysis.pkl`  
**Chart saved:** `docs/cause_analysis.png`

- Loaded and analysed 1.71M fire records from SQLite database
- Top causes found: Debris Burning (25%), Miscellaneous (19%), Arson (16%), Lightning (16%)
- Lightning causes biggest fires — avg 312 acres per fire
- Peak fire months: June and July (avg 226 acres per fire)
- Built prevention tips engine — maps each cause to specific actionable advice
- This powers **Feature F7** of the system

### Step 3C — Improved Risk Scoring Model ✅
**File:** `notebooks/improved_model.py`  
**Model saved:** `models/weights/fire_model_v2.pkl`

- Engineered 3 new features: `heat_dryness`, `wind_rain_ratio`, `drought_score`
- Compared 3 models: Random Forest (AUC 0.938), Gradient Boost (0.909), Logistic Regression (0.937)
- **Best model: Random Forest — AUC 0.938, F1 0.794**
- Built `predict_fire_risk()` function that returns probability + tier + color + action
- This powers **Features F1, F2, F3, F4** of the system

**Current model outputs:**
```
Extreme fire day (42°C, 10% RH, 55km/h wind) → 100.0% → 🔴 HIGH
Medium risk day  (28°C, 45% RH, 20km/h wind) →  61.5% → 🟠 MEDIUM  
Safe day         (18°C, 88% RH,  5km/h wind) →   3.5% → 🟢 LOW
```

---

## ⏳ What Still Needs to Be Done

### Step 4 — FastAPI Backend (Next Step)
Build the REST API that serves model predictions.

**File to create:** `backend/api/main.py`

Endpoints needed:
```
POST /api/predict          ← takes weather input, returns risk score
GET  /api/causes/{region}  ← returns cause breakdown for a region
GET  /api/weather          ← returns live weather for lat/lon
GET  /api/history/{region} ← returns historical fire records
```

How to start:
```bash
pip install fastapi uvicorn
cd backend
# create main.py — see Step 4 instructions
uvicorn api.main:app --reload
```

### Step 5 — Cellular Automata Simulation (Feature F11)
Build the fire spread engine.

**File to create:** `simulation/cellular_automata.py`

Rules:
- Grid = 30m x 30m cells matching prediction raster
- States: 0=Unburned, 1=Burning, 2=Burned, 3=Barrier
- Spread probability = f(wind, slope, fuel, humidity)
- Run 48 iterations = 12 hours, save at 1h/2h/3h/6h/12h

### Step 6 — React Frontend (Features F5, F6, F8, F9, F11, F13)
Build the dashboard.

Components needed:
```
TerrainMap.jsx        ← Leaflet map with green/orange/red overlay (F5)
HeatMap3D.jsx         ← Plotly 3D surface chart (F6)
AIAssistant.jsx       ← Groq API chatbot (F8, F9)
SpreadAnimation.jsx   ← Fire spread animation player (F11)
WeatherWidget.jsx     ← Live weather sidebar (F13)
ContactDirectory.jsx  ← Emergency contacts (F10)
```

### Step 7 — AI Assistant with Groq (Features F8, F9)
Wire up the Groq API chatbot.

```python
# .env already has GROQ_API_KEY
# Use groq Python library
pip install groq
```

### Step 8 — Integration and Testing
- Connect frontend to backend API
- End-to-end test all 13 features
- Performance testing on large rasters

---

## How to Run What Exists Right Now

### 1. Clone and setup
```bash
git clone https://github.com/YOUR_USERNAME/pyrovision.git
cd pyrovision
pip install -r requirements.txt
```

### 2. Download datasets
```bash
# Make sure kaggle.json is at C:\Users\USERNAME\.kaggle\kaggle.json
cd data\raw\viirs
kaggle datasets download -d rtatman/188-million-us-wildfires
Expand-Archive -Path 188-million-us-wildfires.zip -DestinationPath .

cd ..\algeria
kaggle datasets download -d nitinchoudhary012/algerian-forest-fires-dataset
Expand-Archive -Path algerian-forest-fires-dataset.zip -DestinationPath .

cd ..\lulc
kaggle datasets download -d huseynguliyev/landscape-classification
Expand-Archive -Path landscape-classification.zip -DestinationPath .
```

### 3. Verify all datasets are present
```bash
python -c "
import os
folders = {
    'Wildfire SQLite' : 'data/raw/viirs',
    'Algeria CSV'     : 'data/raw/algeria',
    'Weather Data'    : 'data/raw/weather',
    'LULC Images'     : 'data/raw/lulc',
    'DEM Elevation'   : 'data/raw/dem',
}
for name, path in folders.items():
    files = os.listdir(path) if os.path.exists(path) else []
    icon = '✅' if files else '❌'
    print(f'  {icon}  {name:20s}  ->  {len(files)} file(s)')
"
```

### 4. Run data exploration
```bash
cd notebooks
python explore_data.py
```

### 5. Train models
```bash
python train_model.py       # Step 3A — basic model
python cause_analysis.py    # Step 3B — cause analysis
python improved_model.py    # Step 3C — improved model with risk tiers
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
DATABASE_URL=postgresql://localhost:5432/pyrovision
AWS_BUCKET_NAME=pyrovision-data
```

Get Groq API key at: https://console.groq.com

---

## Branching Rules

```
main          ← never push here directly — only merge from dev when stable
  └── dev     ← merge feature branches here after testing
        ├── feature/backend-api
        ├── feature/simulation
        ├── feature/frontend-map
        └── feature/ai-assistant
```

**Always create a new branch before starting any feature:**
```bash
git checkout dev
git checkout -b feature/your-feature-name
```

---

## Model Performance Summary

| Model | F1 Score | AUC | Status |
|---|---|---|---|
| Random Forest v1 | 0.79 | — | Saved as `fire_model_v1.pkl` |
| **Random Forest v2** | **0.794** | **0.938** | **Active — use this one** |
| Gradient Boost | 0.775 | 0.909 | Tested, not saved |
| Logistic Regression | 0.878 | 0.937 | Tested, not saved |

**False Negative Rate: 14.3%** — target is below 10%.  
Needs more training data to improve — will get better when full VIIRS raster data is added in Step 4.

---

## Key Decisions Made

**Why Random Forest over Gradient Boost?**  
Higher AUC (0.938 vs 0.909) and faster inference time — important for hourly re-prediction in Feature F1.

**Why Algeria dataset for training first?**  
Clean, labeled, small — perfect for proving the model works before scaling to 1.88M records.

**Why Groq instead of OpenAI for AI assistant?**  
Faster response time (sub-second vs 2-3 seconds) — critical for the live chatbot in Feature F8.

**Why SQLite for wildfire data?**  
That is how the NASA dataset comes packaged. Will migrate to PostgreSQL + PostGIS in Step 4 for spatial queries.

---

## Contact

Project owner: Abhishek  
Repo: https://github.com/AbhishekYadav65/pyrovision  

_Last updated after Step 3C completion._