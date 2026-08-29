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

        if score >= 45.0:
            risk_escalators.append({
                'Category': cat,
                'Sub_Score': score,
                'Weight': weight,
                'Impact_Score': weighted_contribution
            })
        else:
            risk_mitigators.append({
                'Category': cat,
                'Sub_Score': score,
                'Weight': weight,
                'Impact_Score': weighted_contribution
            })

    risk_escalators = sorted(risk_escalators, key=lambda x: x['Impact_Score'], reverse=True)
    risk_mitigators = sorted(risk_mitigators, key=lambda x: x['Impact_Score'])

    return {
        'Bank': bank_name,
        'Year': year,
        'Overall_Risk_Score': analysis['Assessment']['Risk_Score'],
        'Status': analysis['Assessment']['Status'],
        'Top_Risk_Escalators': risk_escalators[:top_n],
        'Top_Risk_Mitigators': risk_mitigators[:top_n]
    }

def generate_supervisory_recommendation(insights):
    escalators = insights['Top_Risk_Escalators']
    status = insights['Status']

    if not escalators:
        return "المؤشرات العامة مستقرة وضمن الحدود المعيارية المقبولة."

    top_cat = escalators[0]['Category']
    recommendations_map = {
        'Capital': "تعزيز كفاية رأس المال وضبط نمو الأصول المرجحة بالمخاطر وفق متطلبات بازل 3.",
        'Asset Quality': "تشديد معايير منح الائتمان وتدعيم مخصصات خسائر الائتمان المتوقعة لمواجهة ارتفاع التعثر.",
        'Productivity': "رفع كفاءة تشغيل وتوظيف الأصول والموارد مقارنة بمتوسط معيار القطاع.",
        'Profitability': "تحسين هامش الفائدة الصافي وضبط تكلفة التمويل لتعزيز مؤشرات العائد على الأصول.",
        'Efficiency': "ترشيد المصاريف التشغيلية والإدارية وخفض نسبة الأعباء العامة إلى الدخل التشغيلي.",
        'Liquidity': "زيادة نسبة الأصول عالية السيولة وإعادة هيكلة استحقاقات الالتزامات قصيرة الأجل.",
        'Openness': "تفعيل أدوات التحوط لمراكز العملات الأجنبية وإدارة حساسية أسعار الفائدة والصرف."
    }

    primary_advice = recommendations_map.get(top_cat, "متابعة المؤشرات المتراجعة وتطبيق الإجراءات التصحيحية.")
    return f"بناءً على تصنيف [{status}]؛ التوصية هي: {primary_advice}"