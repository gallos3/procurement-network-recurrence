# NOTE: The scripts in this directory are provided for methodological transparency 
# and illustrate the logic used in the study. They are intended for reference 
# and may require adaptation to run in a new environment.

import pandas as pd
import numpy as np

def calculate_hhi(shares):
    """Calculates Herfindahl-Hirschman Index from market shares."""
    return (shares ** 2).sum()

def calculate_vendor_diversity(n_unique, n_total):
    """Calculates ratio of unique suppliers to total contracts."""
    return n_unique / n_total if n_total > 0 else 0

def calculate_authority_metrics(awards_df):
    """
    Illustrative pipeline for Table 3 metrics.
    """
    results = []
    for auth_id, group in awards_df.groupby('authority_id'):
        # Market shares based on contract values
        shares = group.groupby('supplier_id')['value'].sum() / group['value'].sum()
        
        hhi = calculate_hhi(shares)
        diversity = calculate_vendor_diversity(group['supplier_id'].nunique(), len(group))
        
        results.append({
            'authority_id': auth_id,
            'hhi': hhi,
            'vendor_diversity': diversity
        })
    return pd.DataFrame(results)

# These metrics were merged with the XGBoost gains to analyze 
# the relationship between recurrence and market structure.
