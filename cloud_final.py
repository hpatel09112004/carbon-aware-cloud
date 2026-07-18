"""
Carbon-Aware Cloud Computing — Flask API
=========================================
Runs the recommendation engine:
  • ML model prediction (Random Forest)
  • Live Electricity Maps API fallback
  • Priority-aware scoring (carbon / balanced / latency)
  • Free IP geolocation via ip-api.com
  • Grid-zone level emission factors (EPA eGRID 2023)

Run:
    python cloud_final.py
Then open:  http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib, numpy as np, pandas as pd
import requests, os, math, datetime

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

def pth(f): return os.path.join(BASE_DIR, f)

# ── Load artefacts ────────────────────────────────────────────────
print("Loading model artefacts...")
MODEL         = joblib.load(pth('carbon_model.pkl'))
LABEL_ENC     = joblib.load(pth('label_encoder.pkl'))
FEATURE_COLS  = joblib.load(pth('feature_cols.pkl'))
COUNTRY_STATS = pd.read_csv(pth('country_carbon_stats.csv'))
ELEC_MAPS_KEY = os.getenv('ELECTRICITY_MAPS_API_KEY', '')
print(f"  Features: {len(FEATURE_COLS)}  |  Countries: {len(COUNTRY_STATS)}")

# ── Cloud region catalogue (25 regions, grid-zone level) ──────────
# grid_ef = gCO2eq/kWh from EPA eGRID 2023 / IEA 2023
# ci_offset = zone-specific adjustment on top of country ML prediction
CLOUD_REGIONS = {
    'aws-us-east-1':        {'provider':'AWS',   'name':'US East (N. Virginia)',     'lat':38.89, 'lng':-77.03, 'country':'United States', 'grid_zone':'US-MIDA-PJM',    'grid_ef':368, 'ci_offset':+3.2},
    'aws-us-east-2':        {'provider':'AWS',   'name':'US East (Ohio)',             'lat':39.96, 'lng':-82.99, 'country':'United States', 'grid_zone':'US-MIDA-PJM',    'grid_ef':380, 'ci_offset':+5.1},
    'aws-us-west-2':        {'provider':'AWS',   'name':'US West (Oregon)',           'lat':45.52, 'lng':-122.67,'country':'United States', 'grid_zone':'US-NW-PACW',     'grid_ef':132, 'ci_offset':-18.4},
    'aws-ca-central-1':     {'provider':'AWS',   'name':'Canada (Central)',           'lat':45.42, 'lng':-75.69, 'country':'Canada',        'grid_zone':'CA-ON',          'grid_ef':40,  'ci_offset':0.0},
    'aws-eu-west-1':        {'provider':'AWS',   'name':'EU West (Ireland)',          'lat':53.41, 'lng':-8.24,  'country':'Ireland',       'grid_zone':'IE',             'grid_ef':295, 'ci_offset':0.0},
    'aws-eu-central-1':     {'provider':'AWS',   'name':'EU Central (Frankfurt)',     'lat':50.11, 'lng':8.68,   'country':'Germany',       'grid_zone':'DE',             'grid_ef':366, 'ci_offset':0.0},
    'aws-eu-north-1':       {'provider':'AWS',   'name':'EU North (Stockholm)',       'lat':59.33, 'lng':18.06,  'country':'Sweden',        'grid_zone':'SE',             'grid_ef':13,  'ci_offset':0.0},
    'aws-ap-south-1':       {'provider':'AWS',   'name':'Asia Pacific (Mumbai)',      'lat':19.07, 'lng':72.87,  'country':'India',         'grid_zone':'IN-WE',          'grid_ef':708, 'ci_offset':+4.5},
    'aws-ap-northeast-1':   {'provider':'AWS',   'name':'Asia Pacific (Tokyo)',       'lat':35.68, 'lng':139.69, 'country':'Japan',         'grid_zone':'JP-TK',          'grid_ef':463, 'ci_offset':0.0},
    'aws-ap-southeast-1':   {'provider':'AWS',   'name':'Asia Pacific (Singapore)',   'lat':1.35,  'lng':103.82, 'country':'Singapore',     'grid_zone':'SG',             'grid_ef':408, 'ci_offset':0.0},
    'aws-ap-southeast-2':   {'provider':'AWS',   'name':'Asia Pacific (Sydney)',      'lat':-33.86,'lng':151.21, 'country':'Australia',     'grid_zone':'AU-NSW',         'grid_ef':620, 'ci_offset':+2.1},
    'aws-sa-east-1':        {'provider':'AWS',   'name':'South America (São Paulo)',  'lat':-23.55,'lng':-46.63, 'country':'Brazil',        'grid_zone':'BR-CS',          'grid_ef':85,  'ci_offset':0.0},
    'azure-eastus':         {'provider':'Azure', 'name':'East US (Virginia)',         'lat':37.38, 'lng':-79.45, 'country':'United States', 'grid_zone':'US-MIDA-PJM',    'grid_ef':360, 'ci_offset':+2.8},
    'azure-northcentralus': {'provider':'Azure', 'name':'North Central US (Illinois)','lat':41.85, 'lng':-87.65, 'country':'United States', 'grid_zone':'US-MIDW-MISO',   'grid_ef':421, 'ci_offset':+8.3},
    'azure-westus2':        {'provider':'Azure', 'name':'West US 2 (Washington)',     'lat':47.23, 'lng':-119.85,'country':'United States', 'grid_zone':'US-NW-PACW',     'grid_ef':118, 'ci_offset':-19.8},
    'azure-southcentralus': {'provider':'Azure', 'name':'South Central US (Texas)',   'lat':29.76, 'lng':-98.49, 'country':'United States', 'grid_zone':'US-TEX-ERCO',    'grid_ef':430, 'ci_offset':+9.7},
    'azure-westeurope':     {'provider':'Azure', 'name':'West Europe (Netherlands)',  'lat':52.37, 'lng':4.89,   'country':'Netherlands',   'grid_zone':'NL',             'grid_ef':296, 'ci_offset':0.0},
    'azure-centralindia':   {'provider':'Azure', 'name':'Central India (Pune)',       'lat':18.52, 'lng':73.86,  'country':'India',         'grid_zone':'IN-WE',          'grid_ef':714, 'ci_offset':+5.2},
    'gcp-us-central1':      {'provider':'GCP',   'name':'US Central (Iowa)',          'lat':41.59, 'lng':-93.62, 'country':'United States', 'grid_zone':'US-MIDW-MISO',   'grid_ef':392, 'ci_offset':+5.8},
    'gcp-us-east1':         {'provider':'GCP',   'name':'US East (South Carolina)',   'lat':33.19, 'lng':-80.01, 'country':'United States', 'grid_zone':'US-SE-SEPA',     'grid_ef':336, 'ci_offset':-1.2},
    'gcp-us-west1':         {'provider':'GCP',   'name':'US West (Oregon)',           'lat':45.60, 'lng':-121.18,'country':'United States', 'grid_zone':'US-NW-PACW',     'grid_ef':128, 'ci_offset':-18.1},
    'gcp-europe-west1':     {'provider':'GCP',   'name':'Europe West (Belgium)',      'lat':50.85, 'lng':4.35,   'country':'Belgium',       'grid_zone':'BE',             'grid_ef':152, 'ci_offset':0.0},
    'gcp-europe-north1':    {'provider':'GCP',   'name':'Europe North (Finland)',     'lat':60.19, 'lng':24.94,  'country':'Finland',       'grid_zone':'FI',             'grid_ef':72,  'ci_offset':0.0},
    'gcp-asia-south1':      {'provider':'GCP',   'name':'Asia South (Mumbai)',        'lat':19.07, 'lng':72.87,  'country':'India',         'grid_zone':'IN-WE',          'grid_ef':708, 'ci_offset':+4.5},
    'gcp-asia-northeast1':  {'provider':'GCP',   'name':'Asia Northeast (Tokyo)',     'lat':35.68, 'lng':139.69, 'country':'Japan',         'grid_zone':'JP-TK',          'grid_ef':463, 'ci_offset':0.0},
}

# ── Utilities ─────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(max(0, a)))

def get_live_carbon(zone):
    """Real-time carbon intensity from Electricity Maps API."""
    if not ELEC_MAPS_KEY:
        return None, None
    try:
        r = requests.get(
            'https://api.electricitymap.org/v3/carbon-intensity/latest',
            params={'zone': zone},
            headers={'auth-token': ELEC_MAPS_KEY},
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json()
            return d.get('carbonIntensity'), d.get('datetime')
    except Exception:
        pass
    return None, None

def predict_carbon(country, stats_row):
    """Predict carbon intensity using ML model."""
    now = datetime.datetime.now()
    try:
        enc = LABEL_ENC.transform([country])[0]
    except Exception:
        return None
    row = {
        'country_enc':     enc,
        'month':           now.month,
        'year':            now.year,
        'renewable_ratio': float(stats_row.get('avg_renewable_ratio', 0)),
        'fossil_ratio':    float(stats_row.get('avg_fossil_ratio',    0)),
        'total_energy':    float(stats_row.get('total_energy_avg',    0)),
        'Solar':0,'Wind':0,'Hydroelectricity':0,'Nuclear':0,
        'Coal':0,'Oil':0,'Gas':0,'Other sources':0,
    }
    X   = pd.DataFrame([row]).reindex(columns=FEATURE_COLS, fill_value=0)
    out = float(MODEL.predict(X)[0])
    return max(0.0, min(100.0, out))

def compute_score(ci, distance_km, priority):
    """
    Priority-aware combined score.
      carbon   → minimise CI only
      latency  → minimise distance only
      balanced → CI + distance penalty (1 pt per 500 km)
    """
    if priority == 'latency':
        return distance_km
    elif priority == 'carbon':
        return ci
    else:
        return ci + distance_km / 500.0

# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    for path in [
        os.path.join(STATIC_DIR, 'index.html'),
        os.path.join(BASE_DIR,   'index.html'),
    ]:
        if os.path.exists(path):
            return send_file(path)
    return "<h2>index.html not found</h2><p>Place index.html next to cloud_final.py</p>", 404


@app.route('/api/geolocate', methods=['GET'])
def geolocate():
    """Free IP geolocation — ip-api.com (45 req/min, no key needed)."""
    caller = (request.headers.get('X-Forwarded-For','').split(',')[0].strip()
               or request.remote_addr)
    if caller in ('127.0.0.1','::1','localhost'):
        return jsonify({'status':'local','lat':None,'lng':None,'city':None,'country':None})
    try:
        d = requests.get(f'http://ip-api.com/json/{caller}',
                         params={'fields':'status,lat,lon,city,country'}, timeout=5).json()
        if d.get('status') == 'success':
            return jsonify({'status':'success','lat':d['lat'],'lng':d['lon'],
                            'city':d.get('city'),'country':d.get('country')})
    except Exception as e:
        return jsonify({'status':'error','message':str(e),'lat':None,'lng':None})
    return jsonify({'status':'fail','lat':None,'lng':None})


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data         = request.json
    user_lat     = float(data.get('lat',         20.59))
    user_lng     = float(data.get('lng',         78.96))
    provider     = data.get('provider',          'all')
    priority     = data.get('priority',          'balanced')
    workload_kwh = float(data.get('workload_kwh', 100.0))

    candidates = []

    for rid, region in CLOUD_REGIONS.items():
        if provider != 'all' and not rid.startswith(provider):
            continue

        country     = region['country']
        distance_km = haversine(user_lat, user_lng, region['lat'], region['lng'])
        ci_offset   = region.get('ci_offset', 0.0)
        grid_ef     = region.get('grid_ef',    400)

        stats_row = COUNTRY_STATS[COUNTRY_STATS['country'] == country]
        s = stats_row.iloc[0] if not stats_row.empty else pd.Series()
        renewable_pct = float(s.get('avg_renewable_ratio', 0)) * 100 if not s.empty else 0.0

        # ── Carbon intensity: live → ML → historical ──────────────
        live_ci, live_ts = get_live_carbon(region['grid_zone'])

        if live_ci is not None:
            # Electricity Maps returns gCO2/kWh; convert to 0-100 scale
            ci     = min(live_ci / 9.0, 100.0)
            source = 'live'
            ef_co2 = live_ci
        else:
            ci_ml  = predict_carbon(country, s)
            if ci_ml is None:
                ci_ml  = float(s.get('avg_carbon_intensity', 50)) if not s.empty else 50.0
                source = 'historical'
            else:
                source = 'ml_predicted'
            ci     = max(0.0, min(100.0, ci_ml + ci_offset))
            ef_co2 = grid_ef

        combined_score = compute_score(ci, distance_km, priority)
        co2_kg         = round(workload_kwh * ef_co2 / 1000.0, 3)

        candidates.append({
            'region_id':         rid,
            'provider':          region['provider'],
            'region_name':       region['name'],
            'country':           country,
            'lat':               region['lat'],
            'lng':               region['lng'],
            'carbon_intensity':  round(ci, 2),
            'distance_km':       round(distance_km, 0),
            'distance_penalty':  round(distance_km / 500.0, 2),
            'combined_score':    round(combined_score, 2),
            'renewable_pct':     round(renewable_pct, 1),
            'data_source':       source,
            'live_timestamp':    live_ts,
            'grid_ef_gco2kwh':  grid_ef,
            'co2_kg_per_100kwh': co2_kg,
        })

    candidates.sort(key=lambda x: x['combined_score'])
    top3      = candidates[:3]
    worst_ci  = max(r['carbon_intensity'] for r in candidates)
    worst_co2 = max(r['co2_kg_per_100kwh'] for r in candidates)

    for i, r in enumerate(candidates):
        r['rank'] = i + 1
        r['carbon_savings_pct'] = round((worst_ci - r['carbon_intensity']) / (worst_ci + 1e-9) * 100, 1)
        r['co2_saved_kg']       = round(worst_co2 - r['co2_kg_per_100kwh'], 2)

    return jsonify({
        'top_recommendations': top3,
        'all_recommendations': candidates,
        'all_regions_count':   len(candidates),
        'user_location':       {'lat': user_lat, 'lng': user_lng},
        'priority':            priority,
        'workload_kwh':        workload_kwh,
        'co2_summary': {
            'best_kg_per_100kwh':  round(min(r['co2_kg_per_100kwh'] for r in candidates), 2),
            'worst_kg_per_100kwh': round(worst_co2, 2),
            'max_saving_kg':       round(worst_co2 - min(r['co2_kg_per_100kwh'] for r in candidates), 2),
        },
    })


@app.route('/api/countries', methods=['GET'])
def countries():
    return jsonify(COUNTRY_STATS.sort_values('avg_carbon_intensity').to_dict(orient='records'))


@app.route('/api/model-metrics', methods=['GET'])
def model_metrics():
    return jsonify({
        'model': 'Random Forest Regressor',
        'metrics': {'MAE': 0.216, 'RMSE': 0.465, 'R2': 0.9997, 'CV_R2': 0.999},
        'training_samples': 64000,
        'countries': 51, 'sectors': 8, 'date_range': '2019-2025',
        'comparison': [
            {'model': 'Linear Regression', 'MAE': 2.487, 'RMSE': 3.297, 'R2': 0.9825},
            {'model': 'Decision Tree',     'MAE': 0.518, 'RMSE': 1.098, 'R2': 0.9981},
            {'model': 'Random Forest',     'MAE': 0.216, 'RMSE': 0.465, 'R2': 0.9997},
            {'model': 'Gradient Boosting', 'MAE': 0.247, 'RMSE': 0.463, 'R2': 0.9997},
        ],
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)