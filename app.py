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
from pdf_generator import create_bank_risk_pdf, create_sector_risk_pdf

st.set_page_config(
    page_title="EWM-BR | Early Warning Banking Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stat-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
}
.stat-title { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
.stat-title-ar { font-size: 11px; color: #0284c7; margin-bottom: 4px; }
.stat-value { font-size: 22px; font-weight: bold; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

DICT_AR = {
    "title": "🏦 Early Warning System for Banking Risk (EWM-BR)",
    "title_ar": "🏦 منصة الإنذار المبكر للمخاطر المصرفية (EWM-BR)",
    "subtitle": "Predictive Banking Soundness Evaluation based on CAPPELO Framework (Groups 1–7)",
    "subtitle_ar": "التقييم والاستشراف الذكي لسلامة البنوك استناداً لمحاور CAPPELO السبعة",
    "radar_title": "🎯 CAPPELO Multidimensional Radar",
    "radar_title_ar": "رادار تقييم محاور CAPPELO السبعة",
    "gauge_title": "⏱️ Composite Risk Gauge",
    "gauge_title_ar": "مقياس المخاطر المركب (0-100)",
    "xai_title": "🔍 Explainable AI (XAI) Risk Drivers",
    "xai_title_ar": "تفكيك وتفسير أسباب المخاطر (XAI)",
    "matrix_title": "📋 Detailed Variance Matrix vs Industry Norm",
    "matrix_title_ar": "مصفوفة الانحرافات التفصيلية مقابل معيار الصناعة"
}

RECOMMENDATIONS_MAP = {
    'Capital': {
        'en': "Reinforce capital adequacy buffers and optimize risk-weighted asset growth according to Basel III standards.",
        'ar': "تعزيز كفاية رأس المال وضبط نمو الموجودات المرجحة بالمخاطر وفق متطلبات بازل 3."
    },
    'Asset Quality': {
        'en': "Tighten credit underwriting criteria and reinforce expected credit loss (ECL) provisioning against delinquent loans.",
        'ar': "تشديد معايير منح الائتمان وتدعيم مخصصات خسائر الائتمان المتوقعة لمواجهة التعثر."
    },
    'Productivity': {
        'en': "Improve asset deployment and workforce operational productivity relative to the industry benchmark.",
        'ar': "رفع كفاءة تشغيل وتوظيف الأصول والموارد مقارنة بمتوسط معيار القطاع."
    },
    'Profitability': {
        'en': "Enhance Net Interest Margin (NIM) and review wholesale funding costs to improve Return on Assets (ROA).",
        'ar': "تحسين هامش الفائدة الصافي وضبط تكلفة التمويل لتعزيز مؤشرات العائد على الموجودات."
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
    banks = sorted(df[df['Year'] == year]['Bank'].unique().tolist())
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
    st.title("⚙️ Control Panel | لوحة التحكم")
    show_arabic = st.checkbox("🌐 إظهار الترجمة العربية الفورية (Show Arabic)", value=True)
    st.markdown("---")
    
    all_years = sorted(df_data['Year'].unique().tolist(), reverse=True)
    selected_year = st.selectbox("Fiscal Year | السنة المالية:", all_years)
    
    view_selection = st.radio(
        "Mode | نمط العرض:",
        ["🏢 All-Banks Sector Overview (Default)", "🔍 Single Bank Deep-Dive"],
        index=0
    )
    
    bank_list = sorted(df_data[df_data['Year'] == selected_year]['Bank'].unique().tolist())
    if view_selection == "🔍 Single Bank Deep-Dive":
        selected_bank = st.selectbox("Target Bank | المصرف الخاضع للتقييم:", bank_list)
    else:
        selected_bank = bank_list[0]

comp_df = get_all_banks_comparison(df_data, selected_year)

col_header, col_action = st.columns([4, 2])
with col_header:
    st.title(DICT_AR["title"])
    if show_arabic:
        st.markdown(f"<div style='color:#0284c7; font-size:16px; margin-top:-10px;'>{DICT_AR['title_ar']}</div>", unsafe_allow_html=True)
    st.caption(f"{DICT_AR['subtitle']} | Year: {selected_year}")
    if show_arabic:
        st.markdown(f"<div style='color:#64748b; font-size:12px; margin-top:-6px;'>{DICT_AR['subtitle_ar']}</div>", unsafe_allow_html=True)

with col_action:
    st.markdown("<br>", unsafe_allow_html=True)
    if view_selection == "🏢 All-Banks Sector Overview (Default)":
        sector_pdf_bytes = create_sector_risk_pdf(comp_df, selected_year)
        st.download_button(
            label="📄 Download All-Banks PDF Report",
            data=sector_pdf_bytes,
            file_name=f"EWM_BR_Sector_Report_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        analysis_single, insights_single, _ = get_cached_analysis(df_data, selected_bank, selected_year)
        top_cat = insights_single['Top_Risk_Escalators'][0]['Category'] if insights_single['Top_Risk_Escalators'] else 'Capital'
        adv_obj = RECOMMENDATIONS_MAP.get(top_cat, RECOMMENDATIONS_MAP['Capital'])
        single_pdf_bytes = create_bank_risk_pdf(
            bank_name=selected_bank,
            year=selected_year,
            assessment=analysis_single['Assessment'],
            cat_scores=analysis_single['Category_Scores'],
            insights=insights_single,
            advisory=adv_obj['en']
        )
        st.download_button(
            label=f"📄 Download {selected_bank} PDF",
            data=single_pdf_bytes,
            file_name=f"EWM_BR_{selected_bank}_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("---")

if view_selection == "🏢 All-Banks Sector Overview (Default)":
    k1, k2, k3, k4 = st.columns(4)
    total_banks = len(comp_df)
    high_risk_n = len(comp_df[comp_df['Supervisory Status'].str.contains('High Risk', na=False)])
    watch_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Watch|Early', na=False)])
    stable_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Stable', na=False)])

    with k1:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Monitored Institutions</div>' + (f'<div class="stat-title-ar">المصارف الخاضعة للرقابة</div>' if show_arabic else '') + f'<div class="stat-value">{total_banks} Banks</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-card"><div class="stat-title">High Risk Rating</div>' + (f'<div class="stat-title-ar">مؤسسات عالية المخاطر</div>' if show_arabic else '') + f'<div class="stat-value" style="color:#ef4444;">{high_risk_n}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Watch / Early Warning</div>' + (f'<div class="stat-title-ar">مراقبة / إنذار مبكر</div>' if show_arabic else '') + f'<div class="stat-value" style="color:#f59e0b;">{watch_n}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Stable Rating</div>' + (f'<div class="stat-title-ar">مؤسسات مستقرة</div>' if show_arabic else '') + f'<div class="stat-value" style="color:#10b981;">{stable_n}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(f"📊 Banking Sector Composite Risk Ranking ({selected_year})")
    if show_arabic:
        st.caption("مقارنة وترتيب مستويات المخاطر الشاملة لكافة بنوك القطاع")
        
    fig_bar = px.bar(
        comp_df,
        x='Bank',
        y='Risk Score',
        color='Risk Score',
        color_continuous_scale='RdYlGn_r',
        text='Risk Score',
        labels={'Risk Score': 'Risk Score (0-100)', 'Bank': 'Institution'}
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Early Warning Benchmark (50.00)")
    fig_bar.update_layout(height=380, margin=dict(t=25, b=25))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📋 Cross-Institutional Comparative Matrix (All 8 Banks)")
    if show_arabic:
        st.caption("مصفوفة التقييم المقارن الشاملة للمصارف الثمانية عبر محاور CAPPELO")
        
    st.dataframe(
        comp_df.style.format({
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
        height=320
    )

else:
    analysis, insights, var_df = get_cached_analysis(df_data, selected_bank, selected_year)
    assessment = analysis['Assessment']
    cat_scores = analysis['Category_Scores']
    
    raw_status = assessment['Status']
    status_en = STATUS_TRANSLATIONS.get(raw_status, {}).get('en', raw_status)
    status_ar = STATUS_TRANSLATIONS.get(raw_status, {}).get('ar', raw_status)
    
    top_esc = insights['Top_Risk_Escalators'][0]['Category'] if insights['Top_Risk_Escalators'] else 'Capital'
    adv_obj = RECOMMENDATIONS_MAP.get(top_esc, RECOMMENDATIONS_MAP['Capital'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Target Institution</div>' + (f'<div class="stat-title-ar">المصرف المختار</div>' if show_arabic else '') + f'<div class="stat-value" style="font-size:18px;">{selected_bank} ({selected_year})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Risk Score</div>' + (f'<div class="stat-title-ar">درجة المخاطر الإجمالية</div>' if show_arabic else '') + f'<div class="stat-value">{assessment["Risk_Score"]} / 100</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-title">90D Stress Prob.</div>' + (f'<div class="stat-title-ar">احتمالية الضغط المالي</div>' if show_arabic else '') + f'<div class="stat-value">{assessment["Stress_Probability_90D"]}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Supervisory Status</div>' + (f'<div class="stat-title-ar">التصنيف الرقابي</div>' if show_arabic else '') + f'<div class="stat-value" style="color:{assessment["Color"]}; font-size:18px;">{status_en if not show_arabic else status_ar}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_radar, col_gauge = st.columns([6, 4])
    with col_radar:
        st.subheader(DICT_AR["radar_title"])
        if show_arabic:
            st.caption(DICT_AR["radar_title_ar"])
            
        categories = ['Capital (C)', 'Asset Quality (A)', 'Productivity (P)', 'Profitability (P)', 'Efficiency (E)', 'Liquidity (L)', 'Openness (O)']
        values = [cat_scores['Capital'], cat_scores['Asset Quality'], cat_scores['Productivity'], cat_scores['Profitability'], cat_scores['Efficiency'], cat_scores['Liquidity'], cat_scores['Openness']]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name=selected_bank, line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.25)'))
        fig_radar.add_trace(go.Scatterpolar(r=[40, 35, 30, 35, 30, 35, 30, 40], theta=categories + [categories[0]], fill='toself', name='Industry Norm', line_color='#94a3b8', line_dash='dash', fillcolor='rgba(148, 163, 184, 0.15)'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_gauge:
        st.subheader(DICT_AR["gauge_title"])
        if show_arabic:
            st.caption(DICT_AR["gauge_title_ar"])
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment['Risk_Score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Risk Gauge", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': assessment['Color']},
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.15)'},
                    {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                ],
                'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 50.0}
            }
        ))
        fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader(DICT_AR["xai_title"])
    if show_arabic:
        st.caption(DICT_AR["xai_title_ar"])
        
    col_esc, col_mit = st.columns(2)
    with col_esc:
        st.markdown("##### 🚨 Top Risk Escalators | المحاور الأكثر رفعاً للمخاطر")
        if insights['Top_Risk_Escalators']:
            for esc in insights['Top_Risk_Escalators']:
                st.warning(f"**{esc['Category']}:** Sub-Score = `{esc['Sub_Score']:.2f}` (+`{esc['Impact_Score']:.2f}` pts)")
        else:
            st.success("No critical escalators detected.")
    with col_mit:
        st.markdown("##### 🛡️ Top Risk Mitigators | عوامل الأمان وتخفيف المخاطر")
        if insights['Top_Risk_Mitigators']:
            for mit in insights['Top_Risk_Mitigators']:
                st.success(f"**{mit['Category']}:** Sub-Score = `{mit['Sub_Score']:.2f}`")

    st.info(f"**💡 Supervisory Advisory:** {adv_obj['en']}")
    if show_arabic:
        st.markdown(f"<div style='color:#0369a1; font-weight:500; margin-top:-6px; margin-bottom:12px;'><b>💡 التوصية الرقابية:</b> {adv_obj['ar']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(DICT_AR["matrix_title"])
    if show_arabic:
        st.caption(DICT_AR["matrix_title_ar"])
        
    search_term = st.text_input("Search Indicator | بحث في المؤشرات:", "")
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