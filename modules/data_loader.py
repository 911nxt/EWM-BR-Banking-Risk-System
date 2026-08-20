import os
import glob
import pandas as pd
import numpy as np

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
            xlsx_files = glob.glob(os.path.join(directory, "*.xlsx"))
            soundness_files = [f for f in xlsx_files if "Soundness" in f or "CAPPELO" in f]
            if soundness_files:
                return soundness_files[0]
            if xlsx_files:
                return xlsx_files[0]

    raise FileNotFoundError(f"لم يتم العثور على أي ملف إكسل بصيغة .xlsx داخل المجلد: {project_root}")

def load_cappelo_banks_data(file_path=None):
    if file_path is None:
        file_path = get_default_dataset_path()

    print(f"تم العثور على الملف: {os.path.basename(file_path)}")

    df_raw = pd.read_excel(file_path, sheet_name='CAPPELO Banks', header=None)

    columns_metadata = []
    active_bank_name = None

    for col_idx in range(1, df_raw.shape[1]):
        bank_val = df_raw.iloc[1, col_idx]
        year_val = df_raw.iloc[2, col_idx]

        if pd.notnull(bank_val) and str(bank_val).strip() != '':
            active_bank_name = str(bank_val).strip()

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

    indicator_names = df_raw.iloc[:, 0].values
    extracted_records = []

    for meta in columns_metadata:
        c_index = meta['col_idx']
        bank_name = meta['bank']
        year_num = meta['year']

        record = {
            'Bank': bank_name,
            'Year': year_num
        }

        for r_index, ind_name in enumerate(indicator_names):
            if pd.notnull(ind_name) and str(ind_name).strip() != '':
                cleaned_indicator_name = str(ind_name).strip().replace('\n', ' ')
                raw_cell_value = df_raw.iloc[r_index, c_index]
                try:
                    record[cleaned_indicator_name] = float(raw_cell_value)
                except (ValueError, TypeError):
                    record[cleaned_indicator_name] = np.nan

        extracted_records.append(record)

    return pd.DataFrame(extracted_records)

if __name__ == "__main__":
    df_banks = load_cappelo_banks_data()
    print(f"حجم البيانات: {df_banks.shape[0]} صف و {df_banks.shape[1]} عمود")
    print("قائمة البنوك المكتشفة:")
    for b in df_banks['Bank'].unique():
        print(f" - {b}")
    print(f"السنوات: {df_banks['Year'].unique().tolist()}")