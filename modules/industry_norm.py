import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import gmean

def get_excel_file_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    search_dirs = [project_root, os.path.join(project_root, "data"), current_dir]
    for d in search_dirs:
        if os.path.exists(d):
            files = [f for f in glob.glob(os.path.join(d, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
            if files:
                return max(files, key=os.path.getsize)
    return None

def compute_industry_benchmarks(df, year=None):
    file_path = get_excel_file_path()
    if file_path:
        df_raw = pd.read_excel(file_path, sheet_name='CAPPELO Banks', header=None)
        norm_map = {}
        current_group = None
        for r in range(186, df_raw.shape[0]):
            val = df_raw.iloc[r, 0]
            if pd.notnull(val) and str(val).strip() != '':
                name_str = str(val).strip().replace('\n', ' ')
                if "Group " in name_str:
                    current_group = name_str
                    continue
                if current_group is not None:
                    norm_val = df_raw.iloc[r, 19]
                    try:
                        norm_map[name_str] = float(norm_val)
                    except (ValueError, TypeError):
                        pass
        if norm_map:
            return pd.Series(norm_map, name="Industry_Norm")

    target_df = df[df['Year'] == year] if year is not None else df
    numeric_cols = [c for c in target_df.columns if c not in ['Bank', 'Year']]
    benchmarks = {}
    for col in numeric_cols:
        clean = target_df[col].dropna()
        clean = clean[np.isfinite(clean)]
        if len(clean) == 0:
            benchmarks[col] = 0.0
        elif (clean <= 0).any():
            benchmarks[col] = float(clean.mean())
        else:
            benchmarks[col] = float(gmean(clean))
    return pd.Series(benchmarks, name="Industry_Norm")

def generate_variance_matrix(df, bank_name, year):
    bank_row = df[(df['Bank'] == bank_name) & (df['Year'] == year)]
    if bank_row.empty:
        raise ValueError(f"Bank {bank_name} not found for year {year}")

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