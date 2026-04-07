"""
Carbon-Aware Cloud Computing — Flask API
Uses trained Random Forest model + Electricity Maps API
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, numpy as np, pandas as pd, requests, os, math

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Load model artefacts ─────────────────────────────────────────
MODEL        = joblib.load('carbon_model.pkl')
LABEL_ENC    = joblib.load('label_encoder.pkl')
FEATURE_COLS = joblib.load('feature_cols.pkl')
COUNTRY_STATS = pd.read_csv('country_carbon_stats.csv')

ELECTRICITY_MAPS_API_KEY = os.getenv('ELECTRICITY_MAPS_API_KEY', 'your_api_key')

# ── Cloud region catalogue ────────────────────────────────────────
CLOUD_REGIONS = {
    # AWS
    'aws-us-east-1':      {'provider':'AWS','name':'US East (N. Virginia)',   'lat':37.09,'lng':-95.71,'country':'United States'},
    'aws-us-west-2':      {'provider':'AWS','name':'US West (Oregon)',         'lat':43.80,'lng':-120.55,'country':'United States'},
    'aws-eu-west-1':      {'provider':'AWS','name':'EU West (Ireland)',        'lat':53.41,'lng':-8.24,'country':'Ireland'},
    'aws-eu-central-1':   {'provider':'AWS','name':'EU Central (Frankfurt)',   'lat':50.11,'lng':8.68,'country':'Germany'},
    'aws-ap-south-1':     {'provider':'AWS','name':'Asia Pacific (Mumbai)',    'lat':19.07,'lng':72.87,'country':'India'},
    'aws-ap-northeast-1': {'provider':'AWS','name':'Asia Pacific (Tokyo)',     'lat':35.68,'lng':139.69,'country':'Japan'},
    'aws-ap-southeast-1': {'provider':'AWS','name':'Asia Pacific (Singapore)', 'lat':1.35,'lng':103.82,'country':'Singapore'},
    'aws-ap-southeast-2': {'provider':'AWS','name':'Asia Pacific (Sydney)',    'lat':-33.86,'lng':151.21,'country':'Australia'},
    'aws-sa-east-1':      {'provider':'AWS','name':'South America (São Paulo)','lat':-23.55,'lng':-46.63,'country':'Brazil'},
    'aws-eu-north-1':     {'provider':'AWS','name':'EU North (Stockholm)',     'lat':59.33,'lng':18.06,'country':'Sweden'},
    # Azure
    'azure-eastus':       {'provider':'Azure','name':'East US',               'lat':37.38,'lng':-79.45,'country':'United States'},
    'azure-westeurope':   {'provider':'Azure','name':'West Europe',            'lat':52.37,'lng':4.89,'country':'Netherlands'},
    'azure-southeastasia':{'provider':'Azure','name':'Southeast Asia',         'lat':1.29,'lng':103.85,'country':'Singapore'},
    'azure-centralindia': {'provider':'Azure','name':'Central India',          'lat':18.52,'lng':73.86,'country':'India'},
    # GCP
    'gcp-us-central1':    {'provider':'GCP','name':'US Central (Iowa)',        'lat':41.26,'lng':-95.86,'country':'United States'},
    'gcp-europe-west1':   {'provider':'GCP','name':'Europe West (Belgium)',    'lat':50.85,'lng':4.35,'country':'Belgium'},
    'gcp-asia-south1':    {'provider':'GCP','name':'Asia South (Mumbai)',      'lat':19.07,'lng':72.87,'country':'India'},
    'gcp-europe-north1':  {'provider':'GCP','name':'Europe North (Finland)',   'lat':60.19,'lng':24.94,'country':'Finland'},
}

COUNTRY_TO_ZONE = {
    'United States':'US-CAL-CISO','Germany':'DE','France':'FR','India':'IN-NO',
    'Japan':'JP-TK','Australia':'AU-NSW','Brazil':'BR-CS','Sweden':'SE',
    'Netherlands':'NL','Ireland':'IE','Belgium':'BE','Finland':'FI','Singapore':'SG',
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def get_live_carbon(zone):
    """Fetch real-time carbon intensity from Electricity Maps API."""
    try:
        r = requests.get(
            'https://api.electricitymap.org/v3/carbon-intensity/latest',
            params={'zone': zone},
            headers={'auth-token': ELECTRICITY_MAPS_API_KEY},
            timeout=5
        )
        if r.status_code == 200:
            data = r.json()
            return data.get('carbonIntensity'), data.get('datetime')
    except Exception:
        pass
    return None, None

def predict_carbon(country, month=None):
    """Predict carbon intensity for a country using ML model."""
    import datetime
    if month is None:
        month = datetime.datetime.now().month
    year = datetime.datetime.now().year

    stats = COUNTRY_STATS[COUNTRY_STATS['country'] == country]
    if stats.empty:
        return None
    s = stats.iloc[0]

    try:
        enc = LABEL_ENC.transform([country])[0]
    except Exception:
        return None

    row = {
        'country_enc':      enc,
        'month':            month,
        'year':             year,
        'renewable_ratio':  s['avg_renewable_ratio'],
        'fossil_ratio':     s['avg_fossil_ratio'],
        'total_energy':     s['total_energy_avg'],
        'Solar':            0, 'Wind': 0, 'Hydroelectricity': 0,
        'Nuclear': 0, 'Coal': 0, 'Oil': 0, 'Gas': 0, 'Other sources': 0,
    }
    X = pd.DataFrame([row])
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    pred = float(MODEL.predict(X)[0])
    return max(0, min(100, pred))

# ── API Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_lat  = float(data.get('lat', 20.59))
    user_lng  = float(data.get('lng', 78.96))
    provider  = data.get('provider', 'all')   # aws / azure / gcp / all
    workload  = data.get('workload', 'general')

    recommendations = []

    for region_id, region in CLOUD_REGIONS.items():
        if provider != 'all' and not region_id.startswith(provider):
            continue

        country = region['country']
        distance_km = haversine(user_lat, user_lng, region['lat'], region['lng'])

        # Try live Electricity Maps data first
        zone = COUNTRY_TO_ZONE.get(country)
        live_ci, live_ts = get_live_carbon(zone) if zone else (None, None)

        if live_ci is not None:
            carbon_intensity = live_ci / 10   # gCO2/kWh → 0-100 scale approx
            source = 'live'
        else:
            # Fall back to ML prediction
            carbon_intensity = predict_carbon(country)
            source = 'ml_predicted'
            if carbon_intensity is None:
                # Fall back to stats
                stats = COUNTRY_STATS[COUNTRY_STATS['country'] == country]
                carbon_intensity = float(stats['avg_carbon_intensity'].values[0]) if not stats.empty else 50.0
                source = 'historical'

        # Distance penalty: +1 CI point per 500 km
        distance_penalty = distance_km / 500.0
        combined_score = carbon_intensity + distance_penalty

        stats_row = COUNTRY_STATS[COUNTRY_STATS['country'] == country]
        renewable_pct = float(stats_row['avg_renewable_ratio'].values[0]) * 100 if not stats_row.empty else 0

        recommendations.append({
            'region_id':          region_id,
            'provider':           region['provider'],
            'region_name':        region['name'],
            'country':            country,
            'lat':                region['lat'],
            'lng':                region['lng'],
            'carbon_intensity':   round(carbon_intensity, 2),
            'distance_km':        round(distance_km, 0),
            'distance_penalty':   round(distance_penalty, 2),
            'combined_score':     round(combined_score, 2),
            'renewable_pct':      round(renewable_pct, 1),
            'data_source':        source,
            'live_timestamp':     live_ts,
        })

    recommendations.sort(key=lambda x: x['combined_score'])
    top3 = recommendations[:3]

    # Add rank and carbon savings vs worst
    worst_ci = max(r['carbon_intensity'] for r in recommendations)
    for i, r in enumerate(top3):
        r['rank'] = i + 1
        r['carbon_savings_pct'] = round((worst_ci - r['carbon_intensity']) / (worst_ci + 1e-9) * 100, 1)

    return jsonify({
        'top_recommendations': top3,
        'all_regions_count':   len(recommendations),
        'user_location':       {'lat': user_lat, 'lng': user_lng},
    })

@app.route('/api/countries', methods=['GET'])
def countries():
    data = COUNTRY_STATS.sort_values('avg_carbon_intensity').to_dict(orient='records')
    return jsonify(data)

@app.route('/api/model-metrics', methods=['GET'])
def model_metrics():
    return jsonify({
        'model': 'Random Forest Regressor',
        'metrics': {
            'MAE':   0.346,
            'RMSE':  0.553,
            'R2':    0.9994,
            'CV_R2': 0.9988,
        },
        'training_samples': 64000,
        'countries': 51,
        'sectors':   8,
        'date_range': '2019-2025',
        'comparison': [
            {'model':'Linear Regression', 'MAE':5.329,'RMSE':6.878,'R2':0.9096},
            {'model':'Decision Tree',     'MAE':0.812,'RMSE':1.230,'R2':0.9971},
            {'model':'Random Forest',     'MAE':0.321,'RMSE':0.501,'R2':0.9995},
            {'model':'Gradient Boosting', 'MAE':0.501,'RMSE':0.713,'R2':0.9990},
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
