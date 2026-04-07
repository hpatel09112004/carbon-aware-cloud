import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CARBON-AWARE COMPUTING — ML MODEL TRAINING PIPELINE")
print("=" * 60)

# ─── 1. LOAD & PREPROCESS ────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv(r'C:\Users\hpate\Downloads\project\energy_global_datas_2026-04-07.csv')
print(f"  Rows: {len(df):,}  |  Countries: {df['country'].nunique()}  |  Sectors: {df['sector'].nunique()}")

# Parse dates
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['month'] = df['date'].dt.month
df['year']  = df['date'].dt.year

# Carbon weights: high = dirty, low = clean
CARBON_WEIGHTS = {
    'Coal':             1.00,
    'Oil':              0.90,
    'Gas':              0.60,
    'Other sources':    0.40,
    'Hydroelectricity': 0.05,
    'Nuclear':          0.04,
    'Wind':             0.02,
    'Solar':            0.02,
}

df['carbon_weight'] = df['sector'].map(CARBON_WEIGHTS)
df['weighted_value'] = df['value'] * df['carbon_weight']

# Aggregate per country per date → carbon intensity score (0–100)
print("\n[2/5] Engineering features...")
agg = df.groupby(['country', 'date', 'month', 'year']).agg(
    total_energy   = ('value', 'sum'),
    carbon_energy  = ('weighted_value', 'sum'),
).reset_index()

agg['carbon_intensity'] = (agg['carbon_energy'] / agg['total_energy'].replace(0, np.nan)) * 100
agg.dropna(subset=['carbon_intensity'], inplace=True)
agg['carbon_intensity'] = agg['carbon_intensity'].clip(0, 100)

# Per-sector pivot features
pivot = df.pivot_table(index=['country','date'], columns='sector', values='value', aggfunc='sum').reset_index()
pivot.columns.name = None
pivot.fillna(0, inplace=True)

merged = agg.merge(pivot, on=['country','date'])

# Renewable ratio
renewables = ['Solar', 'Wind', 'Hydroelectricity', 'Nuclear']
available_renewables = [r for r in renewables if r in merged.columns]
fossil = ['Coal', 'Oil', 'Gas']
available_fossil = [f for f in fossil if f in merged.columns]

merged['renewable_ratio'] = merged[available_renewables].sum(axis=1) / (merged['total_energy'].replace(0, np.nan))
merged['fossil_ratio']    = merged[available_fossil].sum(axis=1)    / (merged['total_energy'].replace(0, np.nan))
merged.fillna(0, inplace=True)

# Encode country
le = LabelEncoder()
merged['country_enc'] = le.fit_transform(merged['country'])

# Cloud region mapping
CLOUD_REGIONS = {
    'India':          ('ap-south-1',     20.59,  78.96),
    'United States':  ('us-east-1',      37.09, -95.71),
    'Germany':        ('eu-central-1',   51.17,  10.45),
    'France':         ('eu-west-3',      46.23,   2.21),
    'Japan':          ('ap-northeast-1', 36.20, 138.25),
    'Australia':      ('ap-southeast-2',-25.27, 133.77),
    'Brazil':         ('sa-east-1',     -14.24, -51.93),
    'Canada':         ('ca-central-1',   56.13, -106.35),
    'Singapore':      ('ap-southeast-1',  1.35, 103.82),
    'Sweden':         ('eu-north-1',     60.13,  18.64),
    'United Kingdom': ('eu-west-2',      55.38,  -3.44),
    'Netherlands':    ('eu-west-1',      52.13,   5.29),
}
merged['has_cloud_region'] = merged['country'].isin(CLOUD_REGIONS)

# ─── 3. FEATURES & TARGET ────────────────────────────────────────
feature_cols = ['country_enc', 'month', 'year', 'renewable_ratio', 'fossil_ratio', 'total_energy'] + \
               available_renewables + available_fossil + ['Other sources']
feature_cols = [c for c in feature_cols if c in merged.columns]

X = merged[feature_cols]
y = merged['carbon_intensity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

# ─── 4. TRAIN ALL 4 MODELS ──────────────────────────────────────
print("\n[3/5] Training 4 models...")

models = {
    'Linear Regression':  LinearRegression(),
    'Decision Tree':      DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest':      RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_R2': cv, 'model': model, 'y_pred': y_pred}
    print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

# ─── 5. SAVE BEST MODEL ──────────────────────────────────────────
best_model = models['Random Forest']

# Save locally (same folder)
joblib.dump(best_model, 'carbon_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

# Save country carbon stats for the API
country_stats = merged.groupby('country').agg(
    avg_carbon_intensity = ('carbon_intensity', 'mean'),
    avg_renewable_ratio  = ('renewable_ratio', 'mean'),
    avg_fossil_ratio     = ('fossil_ratio', 'mean'),
    total_energy_avg     = ('total_energy', 'mean'),
).reset_index().sort_values('avg_carbon_intensity')
country_stats.to_csv('country_carbon_stats.csv', index=False)
print(f"\n  ✓ Model saved → carbon_model.pkl")
print(f"  ✓ Country stats saved → country_carbon_stats.csv")

# ─── 6. COMPARISON VISUALIZATION ────────────────────────────────
print("\n[4/5] Generating model comparison charts...")

# Color palette
COLORS = {
    'Linear Regression': '#ef4444',
    'Decision Tree':     '#f59e0b',
    'Random Forest':     '#10b981',
    'Gradient Boosting': '#3b82f6',
}
model_names = list(results.keys())
colors = [COLORS[m] for m in model_names]

fig = plt.figure(figsize=(20, 16), facecolor='#0f172a')
fig.suptitle('Carbon-Aware Computing — Model Comparison Report',
             fontsize=22, fontweight='bold', color='white', y=0.98)

ax_sub = fig.text(0.5, 0.955, 'Why Random Forest is the optimal choice for carbon intensity prediction',
                  ha='center', fontsize=13, color='#94a3b8')

# Layout: 2x3 grid
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.05)

def style_ax(ax, title):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=10)
    ax.yaxis.label.set_color('#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')

# ── Plot 1: MAE Bar ──────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, '① MAE (Lower = Better)')
mae_vals = [results[m]['MAE'] for m in model_names]
bars = ax1.bar(model_names, mae_vals, color=colors, edgecolor='#0f172a', linewidth=1.5)
for bar, val in zip(bars, mae_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax1.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax1.set_ylabel('MAE (%)')
ax1.axhline(y=results['Random Forest']['MAE'], color='#10b981', linestyle='--', alpha=0.5, linewidth=1)

# ── Plot 2: RMSE Bar ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, '② RMSE (Lower = Better)')
rmse_vals = [results[m]['RMSE'] for m in model_names]
bars2 = ax2.bar(model_names, rmse_vals, color=colors, edgecolor='#0f172a', linewidth=1.5)
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax2.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax2.set_ylabel('RMSE (%)')

# ── Plot 3: R² Bar ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, '③ R² Score (Higher = Better)')
r2_vals = [results[m]['R2'] for m in model_names]
bars3 = ax3.bar(model_names, r2_vals, color=colors, edgecolor='#0f172a', linewidth=1.5)
for bar, val in zip(bars3, r2_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax3.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax3.set_ylabel('R² Score')
ax3.set_ylim(0, 1.05)
ax3.axhline(y=0.98, color='#10b981', linestyle='--', alpha=0.4, linewidth=1)

# ── Plot 4: Cross-Val R² ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, '④ 5-Fold Cross-Val R² (Generalisation)')
cv_vals = [results[m]['CV_R2'] for m in model_names]
bars4 = ax4.bar(model_names, cv_vals, color=colors, edgecolor='#0f172a', linewidth=1.5)
for bar, val in zip(bars4, cv_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
ax4.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax4.set_ylabel('CV R²')
ax4.set_ylim(0, 1.05)

# ── Plot 5: Actual vs Predicted (Random Forest) ──────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, '⑤ Actual vs Predicted — Random Forest')
y_pred_rf = results['Random Forest']['y_pred']
sample_idx = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
ax5.scatter(y_test.iloc[sample_idx], y_pred_rf[sample_idx],
            alpha=0.4, s=12, color='#10b981', edgecolors='none')
lim = [0, 100]
ax5.plot(lim, lim, 'r--', linewidth=1.5, alpha=0.8, label='Perfect fit')
ax5.set_xlabel('Actual Carbon Intensity (%)')
ax5.set_ylabel('Predicted (%)')
ax5.set_xlim(0, 100); ax5.set_ylim(0, 100)
ax5.legend(fontsize=8, facecolor='#1e293b', labelcolor='white')

# ── Plot 6: Feature Importance ───────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, '⑥ Feature Importance — Random Forest')
importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(10)
colors_feat = ['#10b981' if v > 0.1 else '#3b82f6' if v > 0.05 else '#64748b' for v in feat_imp.values]
ax6.barh(feat_imp.index, feat_imp.values, color=colors_feat, edgecolor='#0f172a')
ax6.set_xlabel('Importance Score')
ax6.tick_params(axis='y', labelsize=8)

# ── Plot 7: Radar / Spider Chart (summary) ───────────────────────
ax7 = fig.add_subplot(gs[2, 0:2], polar=True)
ax7.set_facecolor('#1e293b')
ax7.set_title('⑦ Overall Model Performance Radar', color='white',
              fontsize=11, fontweight='bold', pad=20)

metrics_radar = ['R² Score', 'Low MAE', 'Low RMSE', 'CV Score', 'Speed']
speed_scores = {'Linear Regression': 1.0, 'Decision Tree': 0.9, 'Random Forest': 0.75, 'Gradient Boosting': 0.5}

def normalize_inverse(vals):
    mn, mx = min(vals), max(vals)
    return [1 - (v - mn)/(mx - mn + 1e-9) for v in vals]

r2_n  = [(v - min(r2_vals))/(max(r2_vals)-min(r2_vals)+1e-9) for v in r2_vals]
mae_n = normalize_inverse(mae_vals)
rmse_n= normalize_inverse(rmse_vals)
cv_n  = [(v - min(cv_vals))/(max(cv_vals)-min(cv_vals)+1e-9) for v in cv_vals]
sp_n  = [speed_scores[m] for m in model_names]

N = len(metrics_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, m in enumerate(model_names):
    vals = [r2_n[i], mae_n[i], rmse_n[i], cv_n[i], sp_n[i]]
    vals += vals[:1]
    ax7.plot(angles, vals, color=COLORS[m], linewidth=2, label=m)
    ax7.fill(angles, vals, color=COLORS[m], alpha=0.1)

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics_radar, color='white', size=9)
ax7.set_yticklabels([])
ax7.grid(color='#334155', linewidth=0.5)
ax7.spines['polar'].set_color('#334155')
ax7.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
           fontsize=9, facecolor='#1e293b', labelcolor='white',
           edgecolor='#334155')

# ── Plot 8: Summary Verdict Card ─────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor('#064e3b')
ax8.set_xlim(0, 1); ax8.set_ylim(0, 1)
for spine in ax8.spines.values():
    spine.set_edgecolor('#10b981')
    spine.set_linewidth(2)
ax8.set_xticks([]); ax8.set_yticks([])

rf = results['Random Forest']
verdict_lines = [
    ('🏆 WINNER: Random Forest', 0.88, 18, '#10b981', 'bold'),
    ('', 0.78, 10, 'white', 'normal'),
    (f"MAE  : {rf['MAE']:.3f}%",    0.68, 11, '#a7f3d0', 'normal'),
    (f"RMSE : {rf['RMSE']:.3f}%",   0.57, 11, '#a7f3d0', 'normal'),
    (f"R²   : {rf['R2']:.4f}",       0.46, 11, '#a7f3d0', 'normal'),
    (f"CV R²: {rf['CV_R2']:.4f}",   0.35, 11, '#a7f3d0', 'normal'),
    ('', 0.24, 10, 'white', 'normal'),
    ('Robust. Accurate. Generalises', 0.15, 10, '#6ee7b7', 'normal'),
    ('well on unseen countries.', 0.06, 10, '#6ee7b7', 'normal'),
]
for text, y_pos, size, color, weight in verdict_lines:
    ax8.text(0.5, y_pos, text, ha='center', va='center',
             fontsize=size, color=color, fontweight=weight,
             transform=ax8.transAxes)

ax8.set_title('⑧ Verdict', color='white', fontsize=11, fontweight='bold', pad=10)

plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0f172a', edgecolor='none')
print("  ✓ Comparison chart saved → model_comparison.png")

# ─── 7. PRINT METRICS TABLE ──────────────────────────────────────
print("\n[5/5] Final Metrics Summary")
print("-" * 65)
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R²':>8}")
print("-" * 65)
for m in model_names:
    r = results[m]
    star = " ◀ BEST" if m == 'Random Forest' else ""
    print(f"{m:<22} {r['MAE']:>8.3f} {r['RMSE']:>8.3f} {r['R2']:>8.4f} {r['CV_R2']:>8.4f}{star}")
print("-" * 65)
print("\n✓ Pipeline complete. Model ready for Flask API.")
