# AgriAI — Precision Agriculture ML Platform

An AI-powered farming assistant that provides N-P-K fertilizer dosage predictions, seed quality analysis, live weather integration, and IoT sensor simulation.

## Features
- **N-P-K Regression Engine** — Multi-output Random Forest predicting exact nutrient dosages (kg/ha)
- **Linear Programming Cost Optimizer** — Minimizes fertilizer cost using Urea, DAP, MOP
- **Seed Quality Vision** — CNN with Grad-CAM heatmap overlay for quality grading
- **XAI / SHAP Plots** — Explains model predictions with feature importance charts
- **Live Weather Integration** — Open-Meteo API via geolocation
- **IoT Sensor Simulation** — Mock Arduino/RPi edge device endpoint
- **Multi-language Support** — English, Hindi, Marathi + Google Translate widget
- **Role-Based Access Control** — Admin-only API developer portal

## Requirements
- Python 3.10+
- pip

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd agri-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and set your `FLASK_SECRET_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Run the application
```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | ✅ Yes | Cryptographically secure random key for session signing |
| `DATABASE_URL` | ❌ No | SQLite by default: `sqlite:///farm.db` |
| `FLASK_ENV` | ❌ No | Set to `development` to enable debug mode |

## Admin Access
To access the Developer API Portal (`/api/docs`), you need an account with `is_admin = True`.  
Set this manually via a database migration or Flask shell:
```python
flask shell
>>> from src.models import db, User
>>> u = User.query.filter_by(username='your_username').first()
>>> u.is_admin = True
>>> db.session.commit()
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Health check |
| POST | `/api/v1/predict/fertilizer` | Yes | N-P-K prediction |
| POST | `/api/v1/predict/seed` | Yes | Seed quality classification |
| GET | `/api/v1/sensors/read` | Yes | IoT sensor simulation |
| GET | `/api/stats` | No | Crop/fertilizer analytics |
| GET | `/api/docs` | Admin | Developer documentation |

## Security
- CSRF protection on all forms (Flask-WTF)
- Rate limiting: 10 login attempts/minute, 20 predictions/hour
- 5MB file upload limit with extension whitelist (JPG, PNG, WEBP)
- Secrets loaded from `.env`, never hardcoded
- All errors return generic messages to client; full detail logged server-side
