# CarbonCloud — Carbon-Aware Computing Platform
## Complete Deployment Guide

---

## Project Structure
```
carboncloud/
├── app.py                    # Flask backend API
├── train_model.py            # ML training script (run once)
├── requirements.txt
├── carbon_model.pkl          # Trained Random Forest (auto-generated)
├── label_encoder.pkl         # Country label encoder (auto-generated)
├── feature_cols.pkl          # Feature column names (auto-generated)
├── country_carbon_stats.csv  # Aggregated country stats (auto-generated)
├── static/
│   └── index.html            # Frontend website
└── README.md
```

## Consider cloud_final.py as app.py in this ReadMe.md file
---

## Step 1 — Train the Model

```bash
pip install -r requirements.txt
python train_model.py
```
This generates: `carbon_model.pkl`, `label_encoder.pkl`, `feature_cols.pkl`, `country_carbon_stats.csv`

**Model Results:**
| Model             | MAE   | RMSE  | R²     |
|-------------------|-------|-------|--------|
| Linear Regression | 5.293 | 6.848 | 0.9103 |
| Decision Tree     | 0.835 | 1.235 | 0.9971 |
| **Random Forest** | **0.346** | **0.553** | **0.9994** ← CHOSEN |
| Gradient Boosting | 0.678 | 0.960 | 0.9982 |

---

## Step 2 — Set Electricity Maps API Key

```bash
export ELECTRICITY_MAPS_API_KEY=your_key_here
```

Get your key at: https://www.electricitymap.org/api

---

## Step 3 — Run Locally

```bash
python app.py
# Visit: http://localhost:5000
```

---

## Step 4 — Deploy to AWS EC2

```bash
# On your EC2 instance (Ubuntu 22.04):
sudo apt update && sudo apt install -y python3-pip nginx

# Upload your project files
scp -r carboncloud/ ubuntu@your-ec2-ip:~/

# Install dependencies
cd ~/carboncloud
pip3 install -r requirements.txt

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app &

# Or use systemd service (recommended for production):
```

**systemd service file** (`/etc/systemd/system/carboncloud.service`):
```ini
[Unit]
Description=CarbonCloud Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/carboncloud
Environment="ELECTRICITY_MAPS_API_KEY=your_key"
ExecStart=/usr/local/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable carboncloud
sudo systemctl start carboncloud
```

**Nginx config** (`/etc/nginx/sites-available/carboncloud`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Step 5 — Deploy to Render (Free Alternative)

1. Push to GitHub
2. Go to https://render.com → New Web Service
3. Connect your repo
4. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Environment variable: `ELECTRICITY_MAPS_API_KEY=your_key`

---

## API Endpoints

### POST /api/recommend
```json
{
  "lat": 20.59,
  "lng": 78.96,
  "provider": "all",   // aws | azure | gcp | all
  "workload": "general"
}
```

**Response:**
```json
{
  "top_recommendations": [
    {
      "region_id": "aws-eu-north-1",
      "provider": "AWS",
      "region_name": "EU North (Stockholm)",
      "country": "Sweden",
      "carbon_intensity": 4.2,
      "distance_km": 7842,
      "combined_score": 19.88,
      "renewable_pct": 91.3,
      "data_source": "ml_predicted",
      "rank": 1,
      "carbon_savings_pct": 78.4
    }
  ]
}
```

### GET /api/countries
Returns all 51 countries sorted by avg carbon intensity.

### GET /api/model-metrics
Returns model comparison metrics for all 4 models.

---

## Carbon Scoring Formula

```
combined_score = carbon_intensity + (distance_km / 500)
```

- **carbon_intensity**: ML-predicted or live from Electricity Maps (0–100 scale)
- **distance penalty**: 1 point per 500 km from user location
- Lowest combined score = recommended region

## Carbon Weights Used in Training

| Sector           | Weight |
|------------------|--------|
| Coal             | 1.00   |
| Oil              | 0.90   |
| Gas              | 0.60   |
| Other sources    | 0.40   |
| Hydroelectricity | 0.05   |
| Nuclear          | 0.04   |
| Wind             | 0.02   |
| Solar            | 0.02   |
