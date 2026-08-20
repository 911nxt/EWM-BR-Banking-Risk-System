import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "modules")))

from data_loader import load_cappelo_banks_data
from industry_norm import compute_industry_benchmarks, generate_variance_matrix
from risk_engine import analyze_bank, CAPPELO_WEIGHTS
from xai_explainer import explain_bank_risk_factors
from pdf_generator import create_bank_risk_pdf

st.set_page_config(
    page_title="EWM-BR | Early Warning Banking Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DICTIONARY_AR = {
    "title": "🏦 Early Warning System for Banking Risk (EWM-BR)",
    "title_ar": "🏦 منصة الإنذار المبكر للمخاطر المصرفية (EWM-BR)",
    "subtitle": "Predictive Banking Soundness Evaluation based on CAPPELO Framework & Industry Norms",
    "subtitle_ar": "التقييم والاستشراف الذكي لسلامة البنوك استناداً لمنهجية CAPPELO ومعايير الصناعة",
    "tab_single": "📊 Individual Bank Assessment",
    "tab_single_ar": "تقييم المصرف الفردي",
    "tab_compare": "📑 All-Banks Comparative Matrix",
    "tab_compare_ar": "مصفوفة مقارنة جميع البنوك (8 مصارف)",
    "ctrl_panel": "⚙️ Control Panel",
    "ctrl_panel_ar": "لوحة التحكم والمعايير",
    "select_bank": "Select Target Bank:",
    "select_bank_ar": "المصرف الخاضع للتقييم:",
    "select_year": "Financial Year:",
    "select_year_ar": "السنة المالية:",
    "horizon": "Prediction Horizon:",
    "horizon_ar": "الأفق الزمني للتنبؤ:",
    "export_pdf": "📄 Executive PDF Report",
    "export_pdf_ar": "التقرير التنفيذي (PDF)",
    "radar_title": "🎯 CAPPELO Multidimensional Radar",
    "radar_title_ar": "رادار أداء محاور CAPPELO",
    "gauge_title": "⏱️ Composite Risk Gauge",
    "gauge_title_ar": "عداد مقياس المخاطر المركب",
    "xai_title": "🔍 Explainable AI (XAI) Risk Drivers",
    "xai_title_ar": "تفكيك الأسباب والتفسير السببي (XAI)",
    "escalators": "🚨 Top Risk Escalators (Increasing Risk)",
    "escalators_ar": "المحاور الأكثر رفعاً للمخاطر",
    "mitigators": "🛡️ Top Risk Mitigators (Safety Buffers)",
    "mitigators_ar": "عوامل الأمان وتخفيف المخاطر",
    "rec_title": "💡 Supervisory Advisory:",
    "rec_title_ar": "التوصية الرقابية المقترحة:",
    "matrix_title": "📋 Detailed Variance Matrix vs Industry Norm",
    "matrix_title_ar": "مصفوفة الانحرافات التفصيلية مقابل معيار الصناعة",
    "search": "Search Indicator:",
    "search_ar": "بحث سريع في المؤشرات:",
    "comp_summary": "Comparative Risk Ranking (All 8 Banks)",
    "comp_summary_ar": "ترتيب ومقارنة المخاطر لجميع البنوك الثمانية",
    "comp_barchart": "Banking Sector Risk Score Comparison",
    "comp_barchart_ar": "مقارنة درجات المخاطر الإجمالية في القطاع المصرفي"
}

RECOMMENDATIONS_MAP = {
    'Capital': {
        'en': "Reinforce capital adequacy buffers and optimize risk-weighted asset growth according to Basel III standards.",
        'ar': "تعزيز كفاية رأس المال وضبط نمو الأصول المرجحة بالمخاطر وفق متطلبات بازل 3."
    },
    'Asset Quality': {
        'en': "Tighten credit underwriting criteria and reinforce expected credit loss (ECL) provisioning against delinquent loans.",
        'ar': "تشديد معايير منح الائتمان وتدعيم مخصصات خسائر الائتمان المتوقعة لمواجهة ارتفاع التعثر."
    },
    'Productivity': {
        'en': "Improve asset deployment and workforce operational productivity relative to the industry benchmark.",
        'ar': "رفع كفاءة تشغيل وتوظيف الأصول والموارد مقارنة بمتوسط معيار القطاع."
    },
    'Profitability': {
        'en': "Enhance Net Interest Margin (NIM) and review wholesale funding costs to improve Return on Assets (ROA).",
        'ar': "تحسين هامش الفائدة الصافي وضبط تكلفة التمويل لتعزيز مؤشرات العائد على الأصول."
    },
    'Efficiency': {
        'en': "Streamline cost-to-income overheads and reduce non-interest operating expenditure.",
        'ar': "ترشيد المصاريف التشغيلية والإدارية وخفض نسبة الأعباء العامة إلى الدخل التشغيلي."
    },
    'Liquidity': {
        'en': "Increase High-Quality Liquid Assets (HQLA) allocation and diversify short-term maturity profiles.",
        'ar': "زيادة نسبة الأصول عالية السيولة وإعادة هيكلة استحقاقات الالتزامات قصيرة الأجل."
    },
    'Openness': {
        'en': "Hedge structural foreign currency exposures and manage market sensitivity gaps against interest rate swings.",
        'ar': "تفعيل أدوات التحوط لمراكز العملات الأجنبية وإدارة حساسية أسعار الفائدة والصرف."
    }
}

STATUS_TRANSLATIONS = {
    "مستقر (Stable)": {"en": "Stable", "ar": "مستقر"},
    "مراقبة (Watch)": {"en": "Watch", "ar": "تحت المراقبة"},
    "إنذار مبكر (Early Warning)": {"en": "Early Warning", "ar": "إنذار مبكر"},
    "خطر مرتفع (High Risk)": {"en": "High Risk", "ar": "خطر مرتفع"}
}

@st.cache_data(show_spinner=False)
def get_cached_dataset():
    return load_cappelo_banks_data()

@st.cache_data(show_spinner=False)
def get_cached_analysis(df, bank, year):
    analysis_res = analyze_bank(df, bank, year)
    insights_res = explain_bank_risk_factors(df, bank, year)
    var_res = generate_variance_matrix(df, bank, year)
    return analysis_res, insights_res, var_res

@st.cache_data(show_spinner=False)
def get_all_banks_comparison(df, year):
    banks = df[df['Year'] == year]['Bank'].unique().tolist()
    records = []
    for b in banks:
        try:
            res = analyze_bank(df, b, year)
            ast = res['Assessment']
            cats = res['Category_Scores']
            records.append({
                'Bank': b,
                'Risk Score': ast['Risk_Score'],
                '90D Stress Prob (%)': ast['Stress_Probability_90D'],
                'Supervisory Status': ast['Status'],
                'Capital': round(cats['Capital'], 2),
                'Asset Quality': round(cats['Asset Quality'], 2),
                'Productivity': round(cats['Productivity'], 2),
                'Profitability': round(cats['Profitability'], 2),
                'Efficiency': round(cats['Efficiency'], 2),
                'Liquidity': round(cats['Liquidity'], 2),
                'Openness': round(cats['Openness'], 2)
            })
        except Exception:
            continue
    comp_df = pd.DataFrame(records)
    if not comp_df.empty:
        comp_df = comp_df.sort_values(by='Risk Score', ascending=False).reset_index(drop=True)
    return comp_df

df_data = get_cached_dataset()

with st.sidebar:
    st.title("Settings")
    show_arabic = st.checkbox("🌐 إظهار الترجمة العربية الفورية (Show Arabic Subtitles)", value=False)
    st.markdown("---")
    
    st.header(DICTIONARY_AR["ctrl_panel"])
    if show_arabic:
        st.caption(DICTIONARY_AR["ctrl_panel_ar"])
        
    all_years = sorted(df_data['Year'].unique().tolist(), reverse=True)
    selected_year = st.selectbox(DICTIONARY_AR["select_year"], all_years)
    
    bank_list = sorted(df_data[df_data['Year'] == selected_year]['Bank'].unique().tolist())
    selected_bank = st.selectbox(DICTIONARY_AR["select_bank"], bank_list)
    if show_arabic:
        st.caption(f"المصرف المختار: {selected_bank}")
        
    target_horizon = st.selectbox(DICTIONARY_AR["horizon"], ["90 Days (Standard)", "30 Days", "180 Days"])
    st.markdown("---")

analysis, insights, var_df = get_cached_analysis(df_data, selected_bank, selected_year)
comp_df = get_all_banks_comparison(df_data, selected_year)

assessment = analysis['Assessment']
cat_scores = analysis['Category_Scores']

raw_status = assessment['Status']
status_en = STATUS_TRANSLATIONS.get(raw_status, {}).get('en', raw_status)
status_ar = STATUS_TRANSLATIONS.get(raw_status, {}).get('ar', raw_status)

top_esc = insights['Top_Risk_Escalators'][0]['Category'] if insights['Top_Risk_Escalators'] else 'Capital'
rec_data = RECOMMENDATIONS_MAP.get(top_esc, RECOMMENDATIONS_MAP['Capital'])

with st.sidebar:
    st.subheader(DICTIONARY_AR["export_pdf"])
    if show_arabic:
        st.caption(DICTIONARY_AR["export_pdf_ar"])

    if st.button("Generate PDF Report", use_container_width=True):
        with st.spinner("Compiling PDF..."):
            pdf_bytes = create_bank_risk_pdf(
                bank_name=selected_bank,
                year=selected_year,
                assessment=assessment,
                cat_scores=cat_scores,
                insights=insights,
                advisory=rec_data['en']
            )
            st.session_state['ready_pdf'] = pdf_bytes

    if 'ready_pdf' in st.session_state:
        st.download_button(
            label="📥 Download PDF Report",
            data=st.session_state['ready_pdf'],
            file_name=f"EWM_BR_{selected_bank}_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Main Title Section
st.title(DICTIONARY_AR["title"])
if show_arabic:
    st.markdown(f"<div style='color: #0284c7; font-size: 16px; margin-top: -10px; margin-bottom: 8px;'>{DICTIONARY_AR['title_ar']}</div>", unsafe_allow_html=True)

st.caption(DICTIONARY_AR["subtitle"])
if show_arabic:
    st.markdown(f"<div style='color: #64748b; font-size: 13px; margin-top: -8px;'>{DICTIONARY_AR['subtitle_ar']}</div>", unsafe_allow_html=True)
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs([
    DICTIONARY_AR["tab_single"] if not show_arabic else f"{DICTIONARY_AR['tab_single']} | {DICTIONARY_AR['tab_single_ar']}",
    DICTIONARY_AR["tab_compare"] if not show_arabic else f"{DICTIONARY_AR['tab_compare']} | {DICTIONARY_AR['tab_compare_ar']}"
])

with tab1:
    # 4 Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(label="Target Bank & Year", value=str(selected_year), delta=selected_bank)
        if show_arabic:
            st.caption(f"المصرف: {selected_bank} ({selected_year})")

    with c2:
        st.metric(label="Overall Risk Score", value=f"{assessment['Risk_Score']} / 100")
        if show_arabic:
            st.caption("درجة المخاطر الإجمالية")

    with c3:
        st.metric(label="90-Day Stress Probability", value=f"{assessment['Stress_Probability_90D']}%")
        if show_arabic:
            st.caption("احتمالية الضغط المالي خلال 90 يوماً")

    with c4:
        st.metric(label="Supervisory Status", value=status_en)
        if show_arabic:
            st.caption(f"الحالة: {status_ar} | {assessment['Action']}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_radar, col_gauge = st.columns([6, 4])

    with col_radar:
        st.subheader(DICTIONARY_AR["radar_title"])
        if show_arabic:
            st.caption(DICTIONARY_AR["radar_title_ar"])
        
        categories = ['Capital (C)', 'Asset Quality (A)', 'Productivity (P)', 'Profitability (P)', 'Efficiency (E)', 'Liquidity (L)', 'Openness (O)']
        values = [
            cat_scores['Capital'],
            cat_scores['Asset Quality'],
            cat_scores['Productivity'],
            cat_scores['Profitability'],
            cat_scores['Efficiency'],
            cat_scores['Liquidity'],
            cat_scores['Openness']
        ]
        
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        benchmark_values = [40.0, 35.0, 30.0, 35.0, 30.0, 35.0, 30.0]
        benchmark_closed = benchmark_values + [benchmark_values[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=f'{selected_bank}',
            line_color='#2563eb',
            fillcolor='rgba(37, 99, 235, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=benchmark_closed,
            theta=categories_closed,
            fill='toself',
            name='Industry Norm (Benchmark)',
            line_color='#94a3b8',
            line_dash='dash',
            fillcolor='rgba(148, 163, 184, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=360,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_gauge:
        st.subheader(DICTIONARY_AR["gauge_title"])
        if show_arabic:
            st.caption(DICTIONARY_AR["gauge_title_ar"])
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment['Risk_Score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Risk Gauge (0-100)", 'font': {'size': 15}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': assessment['Color']},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.15)'},
                    {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 50.0
                }
            }
        ))
        fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # XAI Drivers Section
    st.subheader(DICTIONARY_AR["xai_title"])
    if show_arabic:
        st.caption(DICTIONARY_AR["xai_title_ar"])

    col_esc, col_mit = st.columns(2)

    with col_esc:
        st.markdown(f"##### {DICTIONARY_AR['escalators']}")
        if show_arabic:
            st.caption(DICTIONARY_AR['escalators_ar'])
        if insights['Top_Risk_Escalators']:
            for esc in insights['Top_Risk_Escalators']:
                st.warning(f"**{esc['Category']}:** Risk Sub-Score = `{esc['Sub_Score']:.2f}` (Weighted Contribution: `+{esc['Impact_Score']:.2f}` pts)")
        else:
            st.success("No severe risk escalators detected.")

    with col_mit:
        st.markdown(f"##### {DICTIONARY_AR['mitigators']}")
        if show_arabic:
            st.caption(DICTIONARY_AR['mitigators_ar'])
        if insights['Top_Risk_Mitigators']:
            for mit in insights['Top_Risk_Mitigators']:
                st.success(f"**{mit['Category']}:** Risk Sub-Score = `{mit['Sub_Score']:.2f}` (Weighted Contribution: `{mit['Impact_Score']:.2f}` pts)")

    # Advisory Box
    st.info(f"**{DICTIONARY_AR['rec_title']}** {rec_data['en']}")
    if show_arabic:
        st.markdown(f"<div style='color: #0369a1; font-weight: 500; margin-top: -6px; margin-bottom: 12px;'><b>{DICTIONARY_AR['rec_title_ar']}</b> {rec_data['ar']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Variance Matrix Table
    st.subheader(DICTIONARY_AR["matrix_title"])
    if show_arabic:
        st.caption(DICTIONARY_AR["matrix_title_ar"])

    search_term = st.text_input(DICTIONARY_AR["search"], "")
    filtered_var_df = var_df
    if search_term:
        filtered_var_df = var_df[var_df['Indicator'].str.contains(search_term, case=False, na=False)]

    st.dataframe(
        filtered_var_df.style.format({
            'Bank_Value': '{:,.2f}',
            'Industry_Norm': '{:,.2f}',
            'Absolute_Diff': '{:,.2f}',
            'Variance_Pct': '{:+.2f}%'
        }),
        use_container_width=True,
        height=300
    )

with tab2:
    st.subheader(DICTIONARY_AR["comp_summary"])
    if show_arabic:
        st.caption(f"{DICTIONARY_AR['comp_summary_ar']} - السنة المالية {selected_year}")

    st.dataframe(
        comp_df.style.background_gradient(subset=['Risk Score', '90D Stress Prob (%)'], cmap='Reds')
                     .format({
                         'Risk Score': '{:.2f}',
                         '90D Stress Prob (%)': '{:.2f}%',
                         'Capital': '{:.1f}',
                         'Asset Quality': '{:.1f}',
                         'Productivity': '{:.1f}',
                         'Profitability': '{:.1f}',
                         'Efficiency': '{:.1f}',
                         'Liquidity': '{:.1f}',
                         'Openness': '{:.1f}'
                     }),
        use_container_width=True,
        height=330
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(DICTIONARY_AR["comp_barchart"])
    if show_arabic:
        st.caption(DICTIONARY_AR["comp_barchart_ar"])

    fig_bar = px.bar(
        comp_df,
        x='Bank',
        y='Risk Score',
        color='Risk Score',
        color_continuous_scale='RdYlGn_r',
        text='Risk Score',
        labels={'Risk Score': 'Composite Risk Score (0-100)', 'Bank': 'Bank Name'}
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Early Warning Threshold (50)")
    fig_bar.update_layout(height=400, xaxis_tickangle=-25, margin=dict(t=30, b=40))
    st.plotly_chart(fig_bar, use_container_width=True)