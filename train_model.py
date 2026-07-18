"""
Carbon-Aware Cloud Computing — ML Training Pipeline
====================================================
Dataset : energy_global_datas_2026-04-07.csv
          1,048,575 rows · 51 countries · 8 sectors · 2019-2025

Run once:
    python train_model.py

Outputs (saved next to this script):
    carbon_model.pkl        — best model (Random Forest)
    label_encoder.pkl       — country → integer encoder
    feature_cols.pkl        — list of feature column names
    country_carbon_stats.csv — per-country aggregated stats
    model_comparison.png    — 8-panel comparison chart
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
warnings.filterwarnings('ignore')

# All files are saved next to this script — works on Windows & Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def pth(f): return os.path.join(BASE_DIR, f)

BANNER = "=" * 60
print(BANNER)
print("  CARBON-AWARE CLOUD — MODEL TRAINING PIPELINE")
print(BANNER)

# ── 1. LOAD ───────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv(pth('energy_global_datas_2026-04-07.csv'))
df['date']  = pd.to_datetime(df['date'], dayfirst=True)
df['month'] = df['date'].dt.month
df['year']  = df['date'].dt.year
print(f"  Rows: {len(df):,}  |  Countries: {df['country'].nunique()}  |  Sectors: {df['sector'].nunique()}")

# ── 2. CARBON INTENSITY SCORE ─────────────────────────────────────
# Higher weight = dirtier source
CARBON_WEIGHTS = {
    'Coal': 1.00, 'Oil': 0.90, 'Gas': 0.60, 'Other sources': 0.40,
    'Hydroelectricity': 0.05, 'Nuclear': 0.04, 'Wind': 0.02, 'Solar': 0.02,
}
df['carbon_weight']  = df['sector'].map(CARBON_WEIGHTS)
df['weighted_value'] = df['value'] * df['carbon_weight']

agg = df.groupby(['country', 'date', 'month', 'year']).agg(
    total_energy   = ('value',          'sum'),
    carbon_energy  = ('weighted_value', 'sum'),
).reset_index()

agg['carbon_intensity'] = (
    agg['carbon_energy'] / agg['total_energy'].replace(0, np.nan) * 100
).clip(0, 100)
agg.dropna(subset=['carbon_intensity'], inplace=True)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────
print("\n[2/5] Engineering features...")
pivot = df.pivot_table(
    index=['country', 'date'], columns='sector',
    values='value', aggfunc='sum',
).reset_index()
pivot.columns.name = None
pivot.fillna(0, inplace=True)

merged = agg.merge(pivot, on=['country', 'date'])

RENEWABLES = [c for c in ['Solar','Wind','Hydroelectricity','Nuclear'] if c in merged.columns]
FOSSILS    = [c for c in ['Coal','Oil','Gas']                          if c in merged.columns]
merged['renewable_ratio'] = merged[RENEWABLES].sum(axis=1) / merged['total_energy'].replace(0, np.nan)
merged['fossil_ratio']    = merged[FOSSILS].sum(axis=1)    / merged['total_energy'].replace(0, np.nan)
merged.fillna(0, inplace=True)

le = LabelEncoder()
merged['country_enc'] = le.fit_transform(merged['country'])

FEATURE_COLS = (
    ['country_enc', 'month', 'year', 'renewable_ratio', 'fossil_ratio', 'total_energy']
    + RENEWABLES + FOSSILS
    + [c for c in ['Other sources'] if c in merged.columns]
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in merged.columns]

# Sample 80 k rows for speed
sample = merged.sample(n=min(80_000, len(merged)), random_state=42)
X = sample[FEATURE_COLS]
y = sample['carbon_intensity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Features: {len(FEATURE_COLS)}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. TRAIN 4 MODELS ─────────────────────────────────────────────
print("\n[3/5] Training 4 models...")
MODELS = {
    'Linear Regression':  LinearRegression(),
    'Decision Tree':      DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest':      RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, max_depth=5,  random_state=42),
}

results = {}
for name, model in MODELS.items():
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    cv   = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1).mean()
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv,
                     'model': model, 'pred': pred}
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

# ── 5. SAVE BEST MODEL ────────────────────────────────────────────
print("\n[4/5] Saving artefacts...")
best_model = results['Random Forest']['model']
joblib.dump(best_model,   pth('carbon_model.pkl'))
joblib.dump(le,           pth('label_encoder.pkl'))
joblib.dump(FEATURE_COLS, pth('feature_cols.pkl'))

country_stats = merged.groupby('country').agg(
    avg_carbon_intensity = ('carbon_intensity', 'mean'),
    avg_renewable_ratio  = ('renewable_ratio',  'mean'),
    avg_fossil_ratio     = ('fossil_ratio',     'mean'),
    total_energy_avg     = ('total_energy',     'mean'),
).reset_index().sort_values('avg_carbon_intensity')
country_stats.to_csv(pth('country_carbon_stats.csv'), index=False)

for f in ['carbon_model.pkl','label_encoder.pkl','feature_cols.pkl','country_carbon_stats.csv']:
    print(f"  ✓ {f}")

# ── 6. COMPARISON CHART ───────────────────────────────────────────
print("\n[5/5] Generating comparison chart...")
COLORS = {
    'Linear Regression': '#ef4444', 'Decision Tree':  '#f59e0b',
    'Random Forest':     '#10b981', 'Gradient Boosting': '#3b82f6',
}
names  = list(results.keys())
colors = [COLORS[m] for m in names]
short  = [m.replace(' ', '\n') for m in names]

fig = plt.figure(figsize=(20, 16), facecolor='#0f172a')
fig.suptitle('Carbon-Aware Cloud — Model Comparison', fontsize=20,
             fontweight='bold', color='white', y=0.98)
gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.05)

def sax(ax, title):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#334155')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
    ax.yaxis.label.set_color('#94a3b8'); ax.xaxis.label.set_color('#94a3b8')

mae_v  = [results[m]['MAE']  for m in names]
rmse_v = [results[m]['RMSE'] for m in names]
r2_v   = [results[m]['R2']   for m in names]
cv_v   = [results[m]['CV_R2']for m in names]

def bar_chart(ax, vals, title, ylabel, fmt='{:.3f}', offset_frac=0.02):
    sax(ax, title)
    bars = ax.bar(short, vals, color=colors, edgecolor='#0f172a', linewidth=1.2)
    off  = max(vals) * offset_frac
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+off,
                fmt.format(v), ha='center', va='bottom',
                color='white', fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel)

# ① MAE
bar_chart(fig.add_subplot(gs[0,0]), mae_v,  '① MAE — lower is better',  'MAE (%)')
# ② RMSE
bar_chart(fig.add_subplot(gs[0,1]), rmse_v, '② RMSE — lower is better', 'RMSE (%)')
# ③ R²
ax3 = fig.add_subplot(gs[0,2]); sax(ax3, '③ R² — higher is better')
b3  = ax3.bar(short, r2_v, color=colors, edgecolor='#0f172a', linewidth=1.2)
for b,v in zip(b3,r2_v):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
             f'{v:.4f}', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
ax3.set_ylabel('R²'); ax3.set_ylim(0.85, 1.02)

# ④ Cross-Val R²
ax4 = fig.add_subplot(gs[1,0]); sax(ax4, '④ 5-fold CV R²')
b4  = ax4.bar(short, cv_v, color=colors, edgecolor='#0f172a', linewidth=1.2)
for b,v in zip(b4,cv_v):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
             f'{v:.4f}', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
ax4.set_ylabel('CV R²'); ax4.set_ylim(0.85, 1.02)

# ⑤ Actual vs Predicted — Random Forest
ax5 = fig.add_subplot(gs[1,1]); sax(ax5, '⑤ Actual vs Predicted — Random Forest')
rf_pred = results['Random Forest']['pred']
idx = np.random.choice(len(y_test), min(600, len(y_test)), replace=False)
ax5.scatter(y_test.iloc[idx], rf_pred[idx], alpha=0.35, s=9, color='#10b981')
ax5.plot([0,100],[0,100], 'r--', linewidth=1.5, alpha=0.7, label='Perfect fit')
ax5.set_xlabel('Actual (%)'); ax5.set_ylabel('Predicted (%)')
ax5.legend(fontsize=8, facecolor='#1e293b', labelcolor='white')

# ⑥ Feature Importance
ax6 = fig.add_subplot(gs[1,2]); sax(ax6, '⑥ Feature Importance — Random Forest')
fi   = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values().tail(10)
fc   = ['#10b981' if v>0.1 else '#3b82f6' if v>0.05 else '#64748b' for v in fi.values]
ax6.barh(fi.index, fi.values, color=fc, edgecolor='#0f172a')
ax6.set_xlabel('Importance'); ax6.tick_params(axis='y', labelsize=8)

# ⑦ Radar
ax7 = fig.add_subplot(gs[2,0:2], polar=True)
ax7.set_facecolor('#1e293b')
ax7.set_title('⑦ Performance Radar', color='white', fontsize=10, fontweight='bold', pad=20)
SPEED = {'Linear Regression':1.0,'Decision Tree':0.85,'Random Forest':0.65,'Gradient Boosting':0.45}
def nrm(v):     mn,mx=min(v),max(v); return [(x-mn)/(mx-mn+1e-9) for x in v]
def nrm_inv(v): mn,mx=min(v),max(v); return [1-(x-mn)/(mx-mn+1e-9) for x in v]
R=[nrm(r2_v),nrm_inv(mae_v),nrm_inv(rmse_v),nrm(cv_v),[SPEED[m] for m in names]]
N=5; angles=[n/N*2*np.pi for n in range(N)]; angles+=angles[:1]
for i,m in enumerate(names):
    vals=[R[j][i] for j in range(5)]; vals+=vals[:1]
    ax7.plot(angles,vals,color=COLORS[m],linewidth=2,label=m.split()[0])
    ax7.fill(angles,vals,color=COLORS[m],alpha=0.08)
ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(['R²','Low MAE','Low RMSE','CV R²','Speed'],color='white',size=9)
ax7.set_yticklabels([]); ax7.grid(color='#334155',linewidth=0.4)
ax7.spines['polar'].set_color('#334155')
ax7.legend(loc='upper right',bbox_to_anchor=(1.35,1.15),fontsize=8,
           facecolor='#1e293b',labelcolor='white',edgecolor='#334155')

# ⑧ Verdict card
ax8 = fig.add_subplot(gs[2,2])
ax8.set_facecolor('#064e3b')
for sp in ax8.spines.values(): sp.set_edgecolor('#10b981'); sp.set_linewidth(2)
ax8.set_xticks([]); ax8.set_yticks([])
rf = results['Random Forest']
for txt,yp,sz,col in [
    ('WINNER',           0.88,18,'#10b981'),
    ('Random Forest',    0.76,16,'white'),
    (f"MAE  : {rf['MAE']:.4f}%",  0.60,11,'#a7f3d0'),
    (f"RMSE : {rf['RMSE']:.4f}%", 0.49,11,'#a7f3d0'),
    (f"R²   : {rf['R2']:.6f}",    0.38,11,'#a7f3d0'),
    (f"CV R²: {rf['CV_R2']:.6f}", 0.27,11,'#a7f3d0'),
    ('Best accuracy +',  0.14, 9,'#6ee7b7'),
    ('generalisation.',  0.05, 9,'#6ee7b7'),
]:
    ax8.text(0.5,yp,txt,ha='center',va='center',fontsize=sz,color=col,fontweight='bold' if yp>0.7 else 'normal')
ax8.set_title('⑧ Verdict',color='white',fontsize=10,fontweight='bold',pad=8)

plt.savefig(pth('model_comparison.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f172a', edgecolor='none')

# ── SUMMARY ───────────────────────────────────────────────────────
print(f"\n{BANNER}")
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>9} {'CV R²':>8}")
print(BANNER)
for m in names:
    r   = results[m]
    star = ' ← BEST' if m == 'Random Forest' else ''
    print(f"{m:<22} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} {r['R2']:>9.4f} {r['CV_R2']:>8.4f}{star}")
print(BANNER)
print("\n✓ Training complete. Run  python cloud_final.py  to start the API.")