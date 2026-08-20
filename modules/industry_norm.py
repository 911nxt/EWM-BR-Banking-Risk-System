import os
import numpy as np
import pandas as pd
from scipy.stats import gmean
from data_loader import load_cappelo_banks_data

def calculate_metric_norm(series):
    clean_series = series.dropna()
    clean_series = clean_series[np.isfinite(clean_series)]

    if len(clean_series) == 0:
        return 0.0

    if (clean_series <= 0).any():
        return float(clean_series.mean())
    else:
        return float(gmean(clean_series))

def compute_industry_benchmarks(df, year=None):
    if year is not None:
        target_df = df[df['Year'] == year]
    else:
        target_df = df

    if target_df.empty:
        raise ValueError(f"No data available for year: {year}")

    numeric_cols = [c for c in target_df.columns if c not in ['Bank', 'Year']]
    benchmarks = {}

    for col in numeric_cols:
        benchmarks[col] = calculate_metric_norm(target_df[col])

    return pd.Series(benchmarks, name=f"Industry_Norm_{year if year else 'All'}")

def generate_variance_matrix(df, bank_name, year):
    bank_row = df[(df['Bank'] == bank_name) & (df['Year'] == year)]
    
    if bank_row.empty:
        raise ValueError(f"Bank '{bank_name}' not found for year {year}")

    bank_series = bank_row.iloc[0]
    norm_series = compute_industry_benchmarks(df, year=year)

    comparison_records = []
    
    for metric in norm_series.index:
        b_val = bank_series.get(metric, np.nan)
        n_val = norm_series.get(metric, np.nan)

        if pd.notnull(b_val) and pd.notnull(n_val):
            abs_diff = b_val - n_val
            pct_change = ((b_val - n_val) / n_val * 100) if n_val != 0 else np.nan
            
            comparison_records.append({
                'Indicator': metric,
                'Bank_Value': b_val,
                'Industry_Norm': n_val,
                'Absolute_Diff': abs_diff,
                'Variance_Pct': pct_change
            })

    return pd.DataFrame(comparison_records)

if __name__ == "__main__":
    df_banks = load_cappelo_banks_data()
    
    sample_year = 2024
    benchmarks_2024 = compute_industry_benchmarks(df_banks, year=sample_year)
    print(f"Computed benchmarks for {len(benchmarks_2024)} indicators in year {sample_year}.")

    sample_bank = "Qatar Al Watani Bank 1964"
    var_matrix = generate_variance_matrix(df_banks, bank_name=sample_bank, year=sample_year)
    print(f"\nVariance Analysis for: {sample_bank} ({sample_year})")
    print(var_matrix.head(5).to_string(index=False))