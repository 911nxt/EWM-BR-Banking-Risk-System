import os
import glob
import string
import numpy as np
import pandas as pd

def get_default_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    search_directories = [
        project_root,
        os.path.join(project_root, "data"),
        current_dir
    ]
    for directory in search_directories:
        if os.path.exists(directory):
            xlsx_files = [f for f in glob.glob(os.path.join(directory, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
            if xlsx_files:
                return max(xlsx_files, key=os.path.getsize)
    raise FileNotFoundError("Excel file not found.")

def load_cappelo_banks_data(file_path=None):
    if file_path is None:
        file_path = get_default_dataset_path()

    df_raw = pd.read_excel(file_path, sheet_name='CAPPELO Banks', header=None)

    columns_metadata = []
    active_bank_name = None
    bank_mapping = {}
    bank_counter = 0

    for col_idx in range(1, df_raw.shape[1] - 1):
        bank_val = df_raw.iloc[1, col_idx]
        year_val = df_raw.iloc[2, col_idx]

        if pd.notnull(bank_val) and str(bank_val).strip() != '':
            raw_name = str(bank_val).strip()
            if raw_name not in bank_mapping:
                bank_mapping[raw_name] = f"Bank {string.ascii_uppercase[bank_counter]}"
                bank_counter += 1
            active_bank_name = bank_mapping[raw_name]

        if pd.notnull(year_val) and active_bank_name is not None:
            try:
                clean_year = int(float(year_val))
                columns_metadata.append({
                    'col_idx': col_idx,
                    'bank': active_bank_name,
                    'year': clean_year
                })
            except ValueError:
                continue

    indicators = []
    current_group = None
    for r in range(186, df_raw.shape[0]):
        val = df_raw.iloc[r, 0]
        if pd.notnull(val) and str(val).strip() != '':
            name_str = str(val).strip().replace('\n', ' ')
            if "Group " in name_str:
                current_group = name_str
                continue
            if current_group is not None:
                indicators.append((r, name_str))

    extracted_records = []
    for meta in columns_metadata:
        c_index = meta['col_idx']
        record = {
            'Bank': meta['bank'],
            'Year': meta['year']
        }
        for r_index, ind_name in indicators:
            cell_val = df_raw.iloc[r_index, c_index]
            try:
                record[ind_name] = float(cell_val)
            except (ValueError, TypeError):
                record[ind_name] = np.nan
        extracted_records.append(record)

    return pd.DataFrame(extracted_records)