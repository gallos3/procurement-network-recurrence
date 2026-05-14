import os
import itertools
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configuration
PARQUET_FILE = "recurrence_feature_matrix.parquet"
RANDOM_STATE = 42

# Subset selection (Consistent with main analysis)
# Representative subset of CPV categories used for sensitivity analysis
# Selected to capture different structural procurement patterns

TARGET_CPVS  = ["33696", "33111", "50000"]
TARGET_YEARS = [2018, 2024]

# Hyperparameter grid
PARAM_GRID = list(itertools.product(
    [4, 6, 8],          
    [0.03, 0.05, 0.10], 
))

def train_eval(X, y, max_depth, learning_rate):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    
    model = XGBClassifier(
        random_state=RANDOM_STATE, 
        tree_method="hist", 
        missing=np.nan,
        n_estimators=600, 
        max_depth=max_depth, 
        learning_rate=learning_rate,
        subsample=0.8, 
        colsample_bytree=0.8
    )
    
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_val)[:, 1]
    auc    = roc_auc_score(y_val, y_prob)

    # Feature Importance (Gain)
    gain_map = model.get_booster().get_score(importance_type='gain')
    pa_g = float(gain_map.get('f0', 0))
    aa_g = float(gain_map.get('f1', 0))
    hf_g = float(gain_map.get('f2', 0))
    
    total = pa_g + aa_g + hf_g
    if total > 0:
        pa_g /= total; aa_g /= total; hf_g /= total
        
    dominant = max([('HF', hf_g), ('PA', pa_g), ('AA', aa_g)], key=lambda x: x[1])[0]
    return auc, hf_g, pa_g, aa_g, dominant

def main():
    if not os.path.exists(PARQUET_FILE):
        print(f"Error: {PARQUET_FILE} not found.")
        return

    print(f"Loading data from {PARQUET_FILE}...")
    df_all = pd.read_parquet(PARQUET_FILE)

    rows = []
    
    for cpv in TARGET_CPVS:
        for year in TARGET_YEARS:
            print(f"Processing CPV {cpv} | Year {year}...")
            
            subset = df_all[(df_all['cpv'] == cpv) & (df_all['base_year'] == year)]
            if subset.empty:
                continue
                
            X = subset[['pa', 'aa', 'hf']].values
            y = subset['label'].astype(int).values
            
            # Baseline execution (6, 0.05)
            base_auc, base_hf, base_pa, base_aa, base_dom = train_eval(X, y, 6, 0.05)
            baseline_vec = np.array([base_hf, base_pa, base_aa])

            for (max_depth, lr) in PARAM_GRID:
                is_baseline = (max_depth == 6 and lr == 0.05)
                auc, hf_g, pa_g, aa_g, dominant = train_eval(X, y, max_depth, lr)
                
                alt_vec = np.array([hf_g, pa_g, aa_g])
                rho, _ = spearmanr(baseline_vec, alt_vec)

                rows.append({
                    "cpv": cpv,
                    "base_year": year,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "is_baseline": is_baseline,
                    "auc": round(auc, 4),
                    "hf_gain": round(hf_g, 4),
                    "pa_gain": round(pa_g, 4),
                    "aa_gain": round(aa_g, 4),
                    "dominant": dominant,
                    "spearman_rho": round(rho, 4),
                })

    df_results = pd.DataFrame(rows)
    df_results.to_csv("hyperparam_sensitivity_results.csv", index=False)
    print("\nSensitivity analysis complete. Results saved to hyperparam_sensitivity_results.csv")

if __name__ == "__main__":
    main()
