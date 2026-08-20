import os
import numpy as np
import pandas as pd
from data_loader import load_cappelo_banks_data
from industry_norm import compute_industry_benchmarks

CAPPELO_WEIGHTS = {
    'Capital': 0.20,
    'Asset Quality': 0.25,
    'Productivity': 0.10,
    'Profitability': 0.15,
    'Efficiency': 0.10,
    'Liquidity': 0.15,
    'Openness': 0.05
}

def extract_metric_val(data_series, search_keys, default_val=0.0):
    for key in search_keys:
        for col in data_series.index:
            if key.lower() in str(col).lower():
                val = data_series[col]
                if pd.notnull(val) and np.isfinite(val):
                    return float(val)
    return default_val

def evaluate_cappelo_categories(bank_series, norm_series):
    scores = {}

    # 1. Capital Adequacy
    b_cap = extract_metric_val(bank_series, ['Capital / Assets Ratio', 'Simple Ratio', 'Capital'], 12.0)
    n_cap = extract_metric_val(norm_series, ['Capital / Assets Ratio', 'Simple Ratio', 'Capital'], 12.0)
    scores['Capital'] = max(0.0, min(100.0, (1 - (b_cap / (n_cap if n_cap > 0 else 1.0))) * 100 + 35.0))

    # 2. Asset Quality
    b_npl = extract_metric_val(bank_series, ['Non Performing Loans', 'NPL', 'Problem'], 2.0)
    n_npl = extract_metric_val(norm_series, ['Non Performing Loans', 'NPL', 'Problem'], 2.0)
    scores['Asset Quality'] = max(0.0, min(100.0, (b_npl / (n_npl if n_npl > 0 else 1.0)) * 40.0))

    # 3. Productivity
    b_assets = extract_metric_val(bank_series, ['Total Assets'], 1.0)
    n_assets = extract_metric_val(norm_series, ['Total Assets'], 1.0)
    scores['Productivity'] = max(0.0, min(100.0, 30.0 + (1 - min(1.5, b_assets / n_assets)) * 20.0))

    # 4. Profitability
    b_roa = extract_metric_val(bank_series, ['Return on Assets', 'ROA'], 1.5)
    n_roa = extract_metric_val(norm_series, ['Return on Assets', 'ROA'], 1.5)
    scores['Profitability'] = max(0.0, min(100.0, (1 - (b_roa / (n_roa if n_roa > 0 else 1.0))) * 100 + 30.0))

    # 5. Efficiency
    b_burden = extract_metric_val(bank_series, ['Burden Ratio', 'Efficiency'], 1.0)
    n_burden = extract_metric_val(norm_series, ['Burden Ratio', 'Efficiency'], 1.0)
    scores['Efficiency'] = max(0.0, min(100.0, (b_burden / (n_burden if n_burden > 0 else 1.0)) * 35.0))

    # 6. Liquidity
    b_cash = extract_metric_val(bank_series, ['Cash in & Cash at Banks', 'Liquid'], 10.0)
    n_cash = extract_metric_val(norm_series, ['Cash in & Cash at Banks', 'Liquid'], 10.0)
    scores['Liquidity'] = max(0.0, min(100.0, (1 - (b_cash / (n_cash if n_cash > 0 else 1.0))) * 100 + 40.0))

    # 7. Openness & Sensitivity
    b_growth = extract_metric_val(bank_series, ['Assets Growth', 'Growth Index'], 1.0)
    scores['Openness'] = max(0.0, min(100.0, abs(b_growth) * 25.0))

    return scores

def classify_bank_risk(risk_score):
    prob_stress = min(0.99, max(0.01, risk_score / 100.0))

    if prob_stress < 0.25:
        status = "مستقر (Stable)"
        action = "متابعة دورية مستمرة"
        color = "#10b981"
    elif prob_stress < 0.50:
        status = "مراقبة (Watch)"
        action = "متابعة المؤشرات المتدهورة"
        color = "#f59e0b"
    elif prob_stress < 0.75:
        status = "إنذار مبكر (Early Warning)"
        action = "تحليل تفصيلي وتدخل وقائي"
        color = "#f97316"
    else:
        status = "خطر مرتفع (High Risk)"
        action = "مراجعة عاجلة وإجراءات رقابية فورية"
        color = "#ef4444"

    return {
        'Risk_Score': round(risk_score, 2),
        'Stress_Probability_90D': round(prob_stress * 100, 2),
        'Status': status,
        'Action': action,
        'Color': color
    }

def analyze_bank(df, bank_name, year):
    bank_row = df[(df['Bank'] == bank_name) & (df['Year'] == year)]
    if bank_row.empty:
        raise ValueError(f"No records for {bank_name} in {year}")

    bank_series = bank_row.iloc[0]
    norm_series = compute_industry_benchmarks(df, year=year)

    cat_scores = evaluate_cappelo_categories(bank_series, norm_series)
    overall_score = sum(cat_scores[cat] * CAPPELO_WEIGHTS[cat] for cat in CAPPELO_WEIGHTS)
    classification = classify_bank_risk(overall_score)

    return {
        'Bank': bank_name,
        'Year': year,
        'Category_Scores': cat_scores,
        'Assessment': classification
    }

if __name__ == "__main__":
    df_banks = load_cappelo_banks_data()
    test_bank = "Qatar Al Watani Bank 1964"
    test_year = 2024

    result = analyze_bank(df_banks, test_bank, test_year)
    print(f"\n--- Assessment Results for: {result['Bank']} ({result['Year']}) ---")
    print(f"Overall Risk Score: {result['Assessment']['Risk_Score']} / 100")
    print(f"90-Day Stress Probability: {result['Assessment']['Stress_Probability_90D']}%")
    print(f"Status: {result['Assessment']['Status']}")
    print(f"Action: {result['Assessment']['Action']}")
    print("\nCAPPELO Category Sub-Scores:")
    for cat, score in result['Category_Scores'].items():
        print(f" - {cat}: {score:.2f}")