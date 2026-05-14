import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

# Configuration
PARQUET_FILE = "recurrence_feature_matrix.parquet"
RANDOM_STATE = 42
RATIOS = [0.1, 0.5, 1.0]

def train_and_evaluate():
    if not os.path.exists(PARQUET_FILE):
        print(f"Error: {PARQUET_FILE} not found.")
        return

    print(f"Loading data from {PARQUET_FILE}...")
    df_all = pd.read_parquet(PARQUET_FILE)
    df_all = df_all.dropna(subset=['pa', 'aa', 'hf', 'label'])
    
    cpvs = sorted(df_all['cpv'].unique())
    years = sorted(df_all['base_year'].unique())
    
    all_results = []

    print(f"🚀 Running Analysis for Ratios {RATIOS} across {len(cpvs)} CPVs...")

    for ratio in RATIOS:
        print(f"\n--- Testing Negative Ratio: {ratio} ---")
        for cpv in cpvs:
            for year in years:
                # Filter CPV and Year
                subset_cpv_year = df_all[(df_all['cpv'] == cpv) & (df_all['base_year'] == year)]
                
                positives = subset_cpv_year[subset_cpv_year['label'] == 1]
                negatives = subset_cpv_year[subset_cpv_year['label'] == 0]
                
                if len(positives) < 10 or len(negatives) < 10:
                    continue

                # Downsample negatives to match ratio
                n_neg_target = int(len(positives) * ratio)
                if n_neg_target < 1: n_neg_target = 1
                
                # Ensure we don't try to sample more than we have
                n_neg_final = min(n_neg_target, len(negatives))
                
                neg_sampled = negatives.sample(n=n_neg_final, random_state=RANDOM_STATE)
                subset = pd.concat([positives, neg_sampled])
                
                X = subset[['pa', 'aa', 'hf']]
                y = subset['label'].astype(int)
                
                if len(y.unique()) < 2: continue

                # Split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
                )

                # Train
                model = XGBClassifier(
                    random_state=RANDOM_STATE, tree_method="hist",
                    n_estimators=600, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                # Gains
                gain_map = model.get_booster().get_score(importance_type='gain')
                # Use actual feature names as they appear in the DataFrame
                pa_g = float(gain_map.get('pa', 0.0))
                aa_g = float(gain_map.get('aa', 0.0))
                hf_g = float(gain_map.get('hf', 0.0))
                
                tot = pa_g + aa_g + hf_g
                if tot > 0: 
                    pa_g /= tot
                    aa_g /= tot
                    hf_g /= tot

                # Mean Feature Values (Validation set)
                def mean_s(v): return float(np.mean(v)) if len(v) > 0 else 0.0
                
                pa_m_pos = mean_s(X_val[y_val == 1]['pa'])
                pa_m_neg = mean_s(X_val[y_val == 0]['pa'])
                hf_m_pos = mean_s(X_val[y_val == 1]['hf'])
                hf_m_neg = mean_s(X_val[y_val == 0]['hf'])
                aa_m_pos = mean_s(X_val[y_val == 1]['aa'])
                aa_m_neg = mean_s(X_val[y_val == 0]['aa'])

                all_results.append({
                    "ratio": ratio,
                    "cpv": cpv,
                    "base_year": year,
                    "auc": roc_auc_score(y_val, y_proba),
                    "f1": f1_score(y_val, y_pred, zero_division=0),
                    "precision": precision_score(y_val, y_pred, zero_division=0),
                    "recall": recall_score(y_val, y_pred, zero_division=0),
                    "hf_gain": hf_g, "pa_gain": pa_g, "aa_gain": aa_g,
                    "pa_mean_pos": pa_m_pos, "pa_mean_neg": pa_m_neg,
                    "hf_mean_pos": hf_m_pos, "hf_mean_neg": hf_m_neg,
                    "aa_mean_pos": aa_m_pos, "aa_mean_neg": aa_m_neg,
                    "n_pos": len(positives), "n_neg": n_neg_final
                })

    if all_results:
        res_df = pd.DataFrame(all_results)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_csv = os.path.join(script_dir, "appendix_results_full.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"\n✅ All results for Appendix (A1, A3, A4) saved to: {out_csv}")
        
        # Summary for Table 2/A3
        main_results = res_df[res_df['ratio'] == 1.0]
        print(f"\n📈 Summary for Ratio 1.0: Mean AUC={main_results['auc'].mean():.4f}, Mean F1={main_results['f1'].mean():.4f}")

if __name__ == "__main__":
    train_and_evaluate()
