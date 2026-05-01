# 🏠 House Price Prediction — Full-Stack ML Project

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade machine learning system that predicts residential property
prices using **XGBoost**, served via **FastAPI**, with a **Next.js** dashboard.

---

## 📸 Project Preview

> Dashboard → Form inputs on left, live price prediction on right.
> API → Swagger UI at `http://localhost:8000/docs`

---

## 📁 Project Structure


```
House-Price-Prediction/
│
├── api/                 # FastAPI backend
├── frontend/            # Next.js frontend
├── data/                # Dataset
├── models/              # Trained models
├── notebooks/           # EDA & experiments
├── utils/               # Helper functions
├── run.py               # ML pipeline
├── requirements.txt
└── README.md
```


## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Neha-Joshi05/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full ML pipeline
```bash
python run.py
```
This will:
- ✅ Generate 2,000 housing records
- ✅ Clean and engineer 18 features
- ✅ Train Linear Regression, Random Forest, XGBoost
- ✅ Evaluate all models and save comparison report
- ✅ Generate 8 visualisation charts

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
- API → `http://localhost:8000`
- Docs → `http://localhost:8000/docs`

### 6. Start the Dashboard
```bash
cd frontend
npm install
npm run dev
```
- Dashboard → `http://localhost:3000`

---

## 🤖 Models & Results

| Model | R² Score | MAE | MAPE |
|-------|----------|-----|------|
| Linear Regression | 0.9923 | $11,481 | 4.45% |
| **XGBoost** ⭐ | **0.9845** | **$16,743** | **6.82%** |
| Random Forest | 0.9212 | $37,314 | 17.56% |

> XGBoost is used for inference in the API.

---

## 📊 Features

### Raw Features (12)
| Feature | Description |
|---------|-------------|
| area_sqft | Total built-up area (sq ft) |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| floors | Number of floors |
| age_years | Property age in years |
| garage | Has garage (1/0) |
| pool | Has swimming pool (1/0) |
| furnished | unfurnished / semi-furnished / furnished |
| location_tier | 1=Premium → 4=Budget |
| distance_cbd_km | Distance to city centre (km) |
| school_rating | Nearby school rating (1–10) |
| crime_index | Area crime level (lower = safer) |

### Engineered Features (6)
| Feature | Formula |
|---------|---------|
| bed_bath_ratio | bedrooms / bathrooms |
| total_rooms | bedrooms + bathrooms |
| cbd_score | 1 / (distance_cbd_km + 1) |
| furnished_score | 0 / 1 / 2 ordinal |
| amenity_score | garage×1.5 + pool×2.0 + furnished×1.0 |
| age_group | new / modern / old |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Next.js 16, Tailwind CSS |
| Persistence | Joblib |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Server status |
| POST | `/predict` | Predict house price |
| GET | `/model/info` | Active model details |

### Sample Request
```json
POST /predict
{
  "area_sqft": 2200,
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 2,
  "age_years": 8,
  "garage": 1,
  "pool": 0,
  "furnished": "semi-furnished",
  "location_tier": 2,
  "distance_cbd_km": 12.5,
  "school_rating": 7.5,
  "crime_index": 25.0
}
```

### Sample Response
```json
{
  "predicted_price": 345234.03,
  "model_used": "XGBoost",
  "confidence_range": {
    "low": 310710.63,
    "high": 379757.43
  },
  "input_summary": {
    "area_sqft": 2200,
    "bedrooms": 3,
    "location_tier": 2,
    "furnished": "semi-furnished"
  }
}
```

---

## 👨‍💻 Author

NEHA JOSHI 
GitHub: https:(https://github.com/Neha-Joshi05/House-Price-Prediction.git)
LinkedIn: https://www.linkedin.com/in/neha-joshi-0851a2322?utm_source=share_via&utm_content=profile&utm_medium=member_android
⭐ Star this repo if you found it useful!




