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
@media print {
    section[data-testid="stSidebar"], .stDeployButton, footer, header, #MainMenu {
        display: none !important;
    }
    .main .block-container {
        max-width: 100% !important;
        padding: 0.5rem !important;
    }
}
.stat-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    text-align: center;
}
.stat-title { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 5px; }
.stat-value { font-size: 24px; font-weight: bold; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

RECOMMENDATIONS_MAP = {
    'Capital': "Reinforce capital adequacy buffers and optimize risk-weighted asset growth according to Basel III standards.",
    'Asset Quality': "Tighten credit underwriting criteria and reinforce expected credit loss (ECL) provisioning against delinquent loans.",
    'Productivity': "Improve asset deployment and workforce operational productivity relative to the industry benchmark.",
    'Profitability': "Enhance Net Interest Margin (NIM) and review wholesale funding costs to improve Return on Assets (ROA).",
    'Efficiency': "Streamline cost-to-income overheads and reduce non-interest operating expenditure.",
    'Liquidity': "Increase High-Quality Liquid Assets (HQLA) allocation and diversify short-term maturity profiles.",
    'Openness': "Hedge structural foreign currency exposures and manage market sensitivity gaps against interest rate swings."
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

# Sidebar Setup
with st.sidebar:
    st.title("⚙️ Supervisory Panel")
    all_years = sorted(df_data['Year'].unique().tolist(), reverse=True)
    selected_year = st.selectbox("Fiscal Year:", all_years)
    
    st.markdown("---")
    view_selection = st.radio(
        "Navigation Mode:",
        ["🏢 All-Banks Sector Overview (Default)", "🔍 Single Bank Deep-Dive"],
        index=0
    )
    
    bank_list = sorted(df_data[df_data['Year'] == selected_year]['Bank'].unique().tolist())
    if view_selection == "🔍 Single Bank Deep-Dive":
        selected_bank = st.selectbox("Select Target Bank:", bank_list)
    else:
        selected_bank = bank_list[0]

comp_df = get_all_banks_comparison(df_data, selected_year)

# PDF Generation in Sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("📄 Executive PDF Export")
    
    if view_selection == "🏢 All-Banks Sector Overview (Default)":
        if st.button("📑 Generate All-Banks PDF Report", use_container_width=True):
            with st.spinner("Generating Sector PDF..."):
                pdf_sector_bytes = create_sector_risk_pdf(comp_df, selected_year)
                st.session_state['sector_pdf'] = pdf_sector_bytes
                
        if 'sector_pdf' in st.session_state:
            st.download_button(
                label="📥 Download Sector PDF Report",
                data=st.session_state['sector_pdf'],
                file_name=f"EWM_BR_Sector_Report_{selected_year}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        analysis_single, insights_single, _ = get_cached_analysis(df_data, selected_bank, selected_year)
        top_cat = insights_single['Top_Risk_Escalators'][0]['Category'] if insights_single['Top_Risk_Escalators'] else 'Capital'
        adv = RECOMMENDATIONS_MAP.get(top_cat, RECOMMENDATIONS_MAP['Capital'])
        
        if st.button(f"📄 Generate {selected_bank} PDF", use_container_width=True):
            with st.spinner("Compiling PDF..."):
                pdf_single_bytes = create_bank_risk_pdf(
                    bank_name=selected_bank,
                    year=selected_year,
                    assessment=analysis_single['Assessment'],
                    cat_scores=analysis_single['Category_Scores'],
                    insights=insights_single,
                    advisory=adv
                )
                st.session_state['single_pdf'] = pdf_single_bytes
                
        if 'single_pdf' in st.session_state:
            st.download_button(
                label=f"📥 Download {selected_bank} PDF",
                data=st.session_state['single_pdf'],
                file_name=f"EWM_BR_{selected_bank}_{selected_year}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# Top Header
col_header, col_print = st.columns([5, 1])
with col_header:
    st.title("🏦 Early Warning System for Banking Risk (EWM-BR)")
    st.caption(f"CAPPELO Framework Core Empirical Analysis (Groups 1–7) | Monitored Sector: {len(bank_list)} Banks | Year: {selected_year}")
with col_print:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🖨️ Print Screen", use_container_width=True)

st.markdown("---")

# Main Content Routing
if view_selection == "🏢 All-Banks Sector Overview (Default)":
    # Sector High-Level KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    total_banks = len(comp_df)
    high_risk_n = len(comp_df[comp_df['Supervisory Status'].str.contains('High Risk', na=False)])
    watch_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Watch|Early', na=False)])
    stable_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Stable', na=False)])

    with k1:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Monitored Institutions</div><div class="stat-value">{total_banks} Banks</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-card"><div class="stat-title">High Risk Rating</div><div class="stat-value" style="color:#ef4444;">{high_risk_n}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Watch / Early Warning</div><div class="stat-value" style="color:#f59e0b;">{watch_n}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Stable Rating</div><div class="stat-value" style="color:#10b981;">{stable_n}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sector Distribution Chart
    st.subheader(f"📊 Banking Sector Composite Risk Ranking ({selected_year})")
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

    # All-Banks Master Table
    st.subheader("📋 Cross-Institutional Comparative Matrix (All 8 Banks)")
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
    # Single Bank View
    analysis, insights, var_df = get_cached_analysis(df_data, selected_bank, selected_year)
    assessment = analysis['Assessment']
    cat_scores = analysis['Category_Scores']
    top_esc = insights['Top_Risk_Escalators'][0]['Category'] if insights['Top_Risk_Escalators'] else 'Capital'
    advisory_text = RECOMMENDATIONS_MAP.get(top_esc, RECOMMENDATIONS_MAP['Capital'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Selected Institution</div><div class="stat-value" style="font-size:20px;">{selected_bank} ({selected_year})</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Risk Score</div><div class="stat-value">{assessment["Risk_Score"]} / 100</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-title">90D Stress Prob.</div><div class="stat-value">{assessment["Stress_Probability_90D"]}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Supervisory Status</div><div class="stat-value" style="color:{assessment["Color"]}; font-size:18px;">{assessment["Status"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_radar, col_gauge = st.columns([6, 4])
    with col_radar:
        st.subheader("🎯 CAPPELO Multidimensional Radar")
        categories = ['Capital (C)', 'Asset Quality (A)', 'Productivity (P)', 'Profitability (P)', 'Efficiency (E)', 'Liquidity (L)', 'Openness (O)']
        values = [cat_scores['Capital'], cat_scores['Asset Quality'], cat_scores['Productivity'], cat_scores['Profitability'], cat_scores['Efficiency'], cat_scores['Liquidity'], cat_scores['Openness']]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', name=selected_bank, line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.25)'))
        fig_radar.add_trace(go.Scatterpolar(r=[40, 35, 30, 35, 30, 35, 30, 40], theta=categories + [categories[0]], fill='toself', name='Industry Norm', line_color='#94a3b8', line_dash='dash', fillcolor='rgba(148, 163, 184, 0.15)'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_gauge:
        st.subheader("⏱️ Composite Risk Gauge")
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

    st.subheader("🔍 Explainable AI (XAI) Risk Drivers")
    col_esc, col_mit = st.columns(2)
    with col_esc:
        st.markdown("##### 🚨 Top Risk Escalators")
        if insights['Top_Risk_Escalators']:
            for esc in insights['Top_Risk_Escalators']:
                st.warning(f"**{esc['Category']}:** Sub-Score = `{esc['Sub_Score']:.2f}` (+`{esc['Impact_Score']:.2f}` pts)")
        else:
            st.success("No critical escalators detected.")
    with col_mit:
        st.markdown("##### 🛡️ Top Risk Mitigators")
        if insights['Top_Risk_Mitigators']:
            for mit in insights['Top_Risk_Mitigators']:
                st.success(f"**{mit['Category']}:** Sub-Score = `{mit['Sub_Score']:.2f}`")

    st.info(f"**💡 Supervisory Advisory:** {advisory_text}")

    st.markdown("---")
    st.subheader("📋 Detailed Variance Matrix vs Industry Norm")
    search_term = st.text_input("Search Indicator:", "")
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