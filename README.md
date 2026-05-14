## Recurrence as a Governance Signal: Diagnostic Network Metrics for Public Procurement Oversight in Greece

This repository contains the replication code and processed dataset for the research paper:

**Fountoukidis, I.G., Dafli, E.L., Antoniou, I.E., and Varsakelis, N.C. (2026).  
"Recurrence as a Governance Signal: Diagnostic Network Metrics for Public Procurement Oversight in Greece."**

---

### 📊 Overview

This repository provides a replication pipeline for the empirical analysis presented in the paper, including:

- Predictive modelling of buyer–supplier recurrence using XGBoost  
- Feature engineering based on network metrics (HF, PA, AA)  
- Robustness checks across negative sampling ratios  
- Sensitivity analysis of hyperparameters  
- Authority-level market structure indicators (HHI, vendor diversity)

---

### 📂 Repository Structure

- `train_from_parquet.py`: Main modelling and evaluation script  
- `hyperparam_sensitivity.py`: Hyperparameter robustness analysis  
- `feature_engineering.py`: Reference implementation of feature extraction logic (Neo4j)  
- `authority_stats_extraction.py`: Authority-level market structure metrics  
- `recurrence_feature_matrix.parquet`: Processed feature matrix used in the analysis  
- `requirements.txt`: Python dependencies  

---

### ⚠️ Reproducibility Scope

This repository reproduces the modelling, evaluation, and diagnostic analysis presented in the paper.

The following components are not fully reproducible without access to the original infrastructure:

- Raw data extraction from KIMDIS  
- Full Neo4j graph database construction  
- End-to-end pipeline from raw procurement data to the feature matrix  

The provided dataset corresponds to the final feature matrix used for model training and evaluation.

---

### ⚙️ Methodological Notes

- **Temporal design**: Features are computed using data up to year *t* to predict contract awards in *t+1*  
- **Sampling**: Negative sampling ratios of 0.1, 0.5, and 1.0  
- **Coverage**: 12 CPV categories analysed in the paper  
- **Subset analysis**: Selected CPVs used for sensitivity testing  

---

### 📦 Data & Results Availability

The processed dataset and full set of model outputs used in this study are available via Zenodo:

👉 (https://doi.org/10.5281/zenodo.20175804)

⚠️ The dataset is not included in this repository.  
Please download the file `recurrence_feature_matrix.parquet` from Zenodo and place it in the project root before running the scripts.

The repository includes:

- Processed feature matrix used for model training  
- Performance metrics (AUC, F1, Precision, Recall)  
- Feature importance gains (HF, PA, AA)  
- Positive vs negative score distributions  
- Results across sampling ratios and CPV categories  

This dataset corresponds to the final analytical outputs reported in the study.

---

### 🛠️ Installation

```bash
git clone https://github.com/gallos3/procurement-network-recurrence.git
cd your-repo
pip install -r requirements.txt
python train_from_parquet.py
```

---

### 📖 Citation

If you use this code or dataset, please cite:
Fountoukidis, I.G., Dafli, E.L., Antoniou, I.E., and Varsakelis, N.C. (2026).
"Recurrence as a Governance Signal: Diagnostic Network Metrics for Public Procurement Oversight in Greece."

---

### 📜 License
MIT License
