import os
import pandas as pd
import numpy as np
from data_loader import load_cappelo_banks_data
from industry_norm import generate_variance_matrix
from risk_engine import analyze_bank, CAPPELO_WEIGHTS

def explain_bank_risk_factors(df, bank_name, year, top_n=3):
    analysis = analyze_bank(df, bank_name, year)
    cat_scores = analysis['Category_Scores']
    var_matrix = generate_variance_matrix(df, bank_name, year)

    risk_escalators = []
    risk_mitigators = []

    for cat, score in cat_scores.items():
        weight = CAPPELO_WEIGHTS.get(cat, 0.10)
        weighted_contribution = score * weight

        if score >= 50.0:
            risk_escalators.append({
                'Category': cat,
                'Sub_Score': score,
                'Weight': weight,
                'Impact_Score': weighted_contribution,
                'Impact_Type': 'Escalator (رفع المخاطر)'
            })
        else:
            risk_mitigators.append({
                'Category': cat,
                'Sub_Score': score,
                'Weight': weight,
                'Impact_Score': weighted_contribution,
                'Impact_Type': 'Mitigator (تخفيض المخاطر)'
            })

    risk_escalators = sorted(risk_escalators, key=lambda x: x['Impact_Score'], reverse=True)
    risk_mitigators = sorted(risk_mitigators, key=lambda x: x['Impact_Score'])

    insights = {
        'Bank': bank_name,
        'Year': year,
        'Overall_Risk_Score': analysis['Assessment']['Risk_Score'],
        'Status': analysis['Assessment']['Status'],
        'Top_Risk_Escalators': risk_escalators[:top_n],
        'Top_Risk_Mitigators': risk_mitigators[:top_n]
    }

    return insights

def generate_supervisory_recommendation(insights):
    escalators = insights['Top_Risk_Escalators']
    status = insights['Status']

    if not escalators:
        return "المؤشرات العامة مستقرة وضمن الحدود المعيارية المقبولة. يُوصى بالاستمرار في خطة المتابعة الدورية العادية."

    top_cat = escalators[0]['Category']
    
    recommendations_map = {
        'Capital': "تعزيز كفاية رأس المال عبر مراجعة توزيعات الأرباح، وضبط نمو الأصول المرجحة بالمخاطر لضمان مطابقة متطلبات بازل 3.",
        'Asset Quality': "تشديد سياسات الائتمان ومتابعة القروض المتأخرة، مع تدعيم مخصصات خسائر الائتمان المتوقعة لمعالجة ارتفاع مؤشرات التعثر.",
        'Productivity': "رفع كفاءة توظيف الموارد البشرية والتقنية لتحسين معدلات العائد والإنتاجية مقارنة بمعيار القطاع.",
        'Profitability': "تحسين هوامش الفائدة الصافية (NIM) وضبط هيكل تكاليف التمويل لتعزيز مؤشرات العائد على الأصول (ROA).",
        'Efficiency': "ترشيد المصاريف التشغيلية والإدارية ومراجعة نسبة الأعباء التشغيلية بالنسبة لإجمالي الموجودات الدخلية.",
        'Liquidity': "زيادة نسبة الأصول عالية السيولة وإعادة هيكلة مصادر التمويل لتقليل احتمالات التعرض لضغط السيولة الفصلي.",
        'Openness': "ضبط الفجوات في مراكز العملات الأجنبية وتفعيل أدوات التحوط للحد من الحساسية لتغيرات أسعار الصرف والفائدة."
    }

    primary_advice = recommendations_map.get(top_cat, "مراجعة المؤشرات المتراجعة مقارنة بمتوسط القطاع واتخاذ تدابير تصحيحية استباقية.")
    return f"بناءً على تصنيف [{status}]؛ التوصية الأولية هي: {primary_advice}"

if __name__ == "__main__":
    df_banks = load_cappelo_banks_data()
    test_bank = "Qatar Al Watani Bank 1964"
    test_year = 2024

    insights = explain_bank_risk_factors(df_banks, test_bank, test_year)
    rec = generate_supervisory_recommendation(insights)

    print(f"\n================ XAI Breakdown: {insights['Bank']} ({insights['Year']}) ================")
    print(f"Overall Risk Score: {insights['Overall_Risk_Score']} / 100 | Status: {insights['Status']}")
    
    print("\n🚨 Top Factors Increasing Risk (محركات رفع الخطر):")
    for esc in insights['Top_Risk_Escalators']:
        print(f" - {esc['Category']}: Sub-Score = {esc['Sub_Score']:.2f} | Weighted Contribution = {esc['Impact_Score']:.2f}")

    print("\n🛡️ Top Factors Mitigating Risk (محركات الأمان وتخفيض الخطر):")
    for mit in insights['Top_Risk_Mitigators']:
        print(f" - {mit['Category']}: Sub-Score = {mit['Sub_Score']:.2f} | Weighted Contribution = {mit['Impact_Score']:.2f}")

    print(f"\n💡 AI Advisory:\n{rec}")