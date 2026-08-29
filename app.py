import os
import sys
import re
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
    page_title="EWM-BR | Early Warning Banking Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Unified Academic UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;
}

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 32px 28px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    border: 1px solid #334155;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 14px;
    background: rgba(37, 99, 235, 0.25);
    border: 1px solid rgba(96, 165, 250, 0.4);
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 12px;
}

.stat-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}
.stat-box:hover {
    border-color: #94a3b8;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.pillar-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.footer-box {
    text-align: center;
    padding: 24px 0;
    margin-top: 40px;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

I18N = {
    'en': {
        'portal_badge': "Regulatory & Empirical Supervisory Framework",
        'hero_title': "Early Warning Model for Banks' Risk (EWM-BR)",
        'hero_subtitle': "Empirical Financial Soundness Evaluation based on CAPPELO Framework (Groups 1–7) across 8 Anonymized Banking Institutions.",
        'sec_overview': "🏛️ 1. CAPPELO Framework & Supervisory Foundations",
        'sec_sector': "📊 2. Banking Sector Macro Risk Assessment",
        'sec_deepdive': "🔍 3. Institutional Deep-Dive Assessment",
        'sec_variance': "📋 4. Indicators Variance Matrix vs. Sector Benchmark",
        'monitored_banks': "Monitored Institutions",
        'cappelo_pillars': "CAPPELO Dimensions",
        'kpis_count': "Standardized KPIs",
        'stress_horizon': "Stress Forecast Horizon",
        'high_risk': "High Risk Exposure",
        'watch_warning': "Watch / Early Warning",
        'stable': "Stable Soundness",
        'risk_score': "Overall Risk Score",
        'stress_prob': "90-Day Stress Probability",
        'supervisory_status': "Supervisory Classification",
        'radar_title': "🎯 CAPPELO Multidimensional Risk Profile",
        'gauge_title': "⏱️ Composite Risk Gauge",
        'xai_title': "🔍 Explainable AI (XAI) Risk Decomposition",
        'escalators': "🚨 Key Risk Escalators (Increasing Vulnerability)",
        'mitigators': "🛡️ Key Risk Mitigators (Safety Buffers)",
        'advisory_title': "💡 Supervisory Directive & Recommendation:",
        'search': "Search KPI / Indicator Name:",
        'download_sector': "📄 Download Master Sector PDF",
        'download_single': "📄 Download Institution PDF",
        'col_bank': "Bank Institution",
        'col_risk': "Risk Score (0-100)",
        'col_prob': "90D Stress Prob.",
        'col_status': "Supervisory Rating",
        'col_cap': "Capital (C)",
        'col_ast': "Asset Quality (A)",
        'col_prd': "Productivity (P)",
        'col_prf': "Profitability (P)",
        'col_eff': "Efficiency (E)",
        'col_liq': "Liquidity (L)",
        'col_opn': "Openness (O)"
    },
    'ar': {
        'portal_badge': "المنظومة القياسية والرقابية لتقييم السلامة المصرفية",
        'hero_title': "منظومة الإنذار المبكر للمخاطر المصرفية (EWM-BR)",
        'hero_subtitle': "التقييم والاستشراف الذكي لسلامة المصارف استناداً لمحاور CAPPELO السبعة (المجموعات 1–7) لـ 8 بنوك مجهلة لضمان الحيادية والسرية.",
        'sec_overview': "🏛️ 1. المرتكزات الرقابية ومحاور نموذج CAPPELO",
        'sec_sector': "📊 2. التقييم الشامل لدرجات مخاطر القطاع المصرفي",
        'sec_deepdive': "🔍 3. التحليل التفصيلي المعمق للمصرف المختار",
        'sec_variance': "📋 4. مصفوفة انحراف المؤشرات المالية عن معيار القطاع",
        'monitored_banks': "المصارف الخاضعة للرقابة",
        'cappelo_pillars': "محاور CAPPELO التحليلية",
        'kpis_count': "مؤشراً مالياً معيارياً",
        'stress_horizon': "أفق التنبؤ بالضغط المالي",
        'high_risk': "مؤسسات عالية المخاطر",
        'watch_warning': "مؤسسات قيد المراقبة / إنذار",
        'stable': "مؤسسات مستقرة",
        'risk_score': "درجة المخاطر المركبة",
        'stress_prob': "احتمالية الضغط المالي (90 يوماً)",
        'supervisory_status': "التصنيف الرقابي",
        'radar_title': "🎯 رادار أبعاد السلامة المصرفية (CAPPELO)",
        'gauge_title': "⏱️ مقياس مؤشر المخاطر التراكمي",
        'xai_title': "🔍 التفسير السببي وتفكيك محركات الخطر (XAI)",
        'escalators': "🚨 المحاور الأكثر رفعاً للمخاطر",
        'mitigators': "🛡️ محركات الأمان وتخفيف المخاطر",
        'advisory_title': "💡 التوجيه والتوصية الرقابية المقترحة:",
        'search': "بحث سريع في المؤشرات المالية:",
        'download_sector': "📄 تحميل تقرير القطاع الشامل (PDF)",
        'download_single': "📄 تحميل تقرير المصرف (PDF)",
        'col_bank': "المصرف",
        'col_risk': "درجة الخطر",
        'col_prob': "احتمال الضغط",
        'col_status': "التصنيف الرقابي",
        'col_cap': "كفاية رأس المال",
        'col_ast': "جودة الأصول",
        'col_prd': "الإنتاجية",
        'col_prf': "الربحية",
        'col_eff': "الكفاءة التشغيلية",
        'col_liq': "السيولة",
        'col_opn': "الحساسية والسوق"
    }
}

RECOMMENDATIONS = {
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

STATUS_MAP = {
    "مستقر (Stable)": {"en": "Stable", "ar": "مستقر"},
    "مراقبة (Watch)": {"en": "Watch", "ar": "تحت المراقبة"},
    "إنذار مبكر (Early Warning)": {"en": "Early Warning", "ar": "إنذار مبكر"},
    "خطر مرتفع (High Risk)": {"en": "High Risk", "ar": "خطر مرتفع"}
}

def clean_indicator_string(text, lang='en'):
    raw = str(text)
    if lang == 'en':
        clean = re.sub(r'\([\s\u0600-\u06FF0-9\-–]*\)', '', raw)
        clean = re.sub(r'[\u0600-\u06FF]', '', clean)
        return re.sub(r'\s+', ' ', clean).strip()
    return raw

@st.cache_data(show_spinner=False)
def get_dataset():
    return load_cappelo_banks_data()

@st.cache_data(show_spinner=False)
def get_analysis(df, bank, year):
    analysis_res = analyze_bank(df, bank, year)
    insights_res = explain_bank_risk_factors(df, bank, year)
    var_res = generate_variance_matrix(df, bank, year)
    return analysis_res, insights_res, var_res

@st.cache_data(show_spinner=False)
def get_sector_data(df, year, lang='en'):
    banks = sorted(df[df['Year'] == year]['Bank'].unique().tolist())
    records = []
    for b in banks:
        try:
            res = analyze_bank(df, b, year)
            ast = res['Assessment']
            cats = res['Category_Scores']
            raw_st = ast['Status']
            st_text = STATUS_MAP.get(raw_st, {}).get(lang, raw_st)
            records.append({
                'Bank': b,
                'Risk Score': ast['Risk_Score'],
                '90D Stress Prob (%)': ast['Stress_Probability_90D'],
                'Supervisory Status': st_text,
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

df_data = get_dataset()

# ================= SIDEBAR CONTROLS =================
with st.sidebar:
    st.markdown("### ⚙️ Supervisory Controls")
    enable_arabic = st.toggle("🌐 إظهار الواجهة باللغة العربية (Arabic)", value=False)
    lang = 'ar' if enable_arabic else 'en'
    T = I18N[lang]

    st.markdown("---")
    all_years = sorted(df_data['Year'].unique().tolist(), reverse=True)
    selected_year = st.selectbox("Fiscal Year | السنة المالية:", all_years)

    bank_list = sorted(df_data[df_data['Year'] == selected_year]['Bank'].unique().tolist())
    selected_bank = st.selectbox("Selected Target Bank | المصرف قيد الفحص:", bank_list)

    st.markdown("---")
    st.markdown("### 📄 Reports Center")
    comp_df_en = get_sector_data(df_data, selected_year, lang='en')
    pdf_sector_bytes = create_sector_risk_pdf(comp_df_en, selected_year)
    st.download_button(
        label=T['download_sector'],
        data=pdf_sector_bytes,
        file_name=f"EWM_BR_Sector_Report_{selected_year}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    analysis_s, insights_s, _ = get_analysis(df_data, selected_bank, selected_year)
    top_c = insights_s['Top_Risk_Escalators'][0]['Category'] if insights_s['Top_Risk_Escalators'] else 'Capital'
    adv_en = RECOMMENDATIONS.get(top_c, RECOMMENDATIONS['Capital'])['en']
    pdf_single_bytes = create_bank_risk_pdf(
        bank_name=selected_bank,
        year=selected_year,
        assessment=analysis_s['Assessment'],
        cat_scores=analysis_s['Category_Scores'],
        insights=insights_s,
        advisory=adv_en
    )
    st.download_button(
        label=f"{T['download_single']} ({selected_bank})",
        data=pdf_single_bytes,
        file_name=f"EWM_BR_{selected_bank}_{selected_year}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# ================= HEADER / HERO BANNER =================
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-badge">🏛️ {T['portal_badge']}</div>
    <h1 style="margin: 0; font-size: 28px; font-weight: 800; line-height: 1.3;">{T['hero_title']}</h1>
    <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 14px; max-width: 900px; line-height: 1.6;">
        {T['hero_subtitle']}
    </p>
</div>
""", unsafe_allow_html=True)

# Top Stat Summary Cards
c_s1, c_s2, c_s3, c_s4 = st.columns(4)
with c_s1:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">{T["monitored_banks"]}</div><div style="font-size:24px; font-weight:800; color:#0f172a; margin-top:4px;">8 Banks</div><div style="font-size:11px; color:#94a3b8;">Anonymized (A–H)</div></div>', unsafe_allow_html=True)
with c_s2:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">{T["cappelo_pillars"]}</div><div style="font-size:24px; font-weight:800; color:#2563eb; margin-top:4px;">7 Pillars</div><div style="font-size:11px; color:#94a3b8;">Groups 1–7</div></div>', unsafe_allow_html=True)
with c_s3:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">{T["kpis_count"]}</div><div style="font-size:24px; font-weight:800; color:#10b981; margin-top:4px;">144 KPIs</div><div style="font-size:11px; color:#94a3b8;">Standard Norms</div></div>', unsafe_allow_html=True)
with c_s4:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">{T["stress_horizon"]}</div><div style="font-size:24px; font-weight:800; color:#6366f1; margin-top:4px;">90 Days</div><div style="font-size:11px; color:#94a3b8;">Quarterly Stress Prob.</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= SECTION 1: CAPPELO PILLARS OVERVIEW =================
with st.expander(T['sec_overview'], expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"""
        <div class="pillar-card">
            <b style="color:#2563eb;">1. Capital Adequacy (C)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'كفاية رأس المال والرافعة المالية ومطابقة بازل 3 لامتصاص الصدمات.' if lang=='ar' else 'Capital buffer, leverage limits, and risk-weighted asset capacity.'}
            </span>
        </div>
        <div class="pillar-card">
            <b style="color:#ef4444;">2. Asset Quality (A)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'جودة المحفظة الائتمانية ورصد القروض غير المنتظمة ومخصصات التعثر.' if lang=='ar' else 'Non-performing loan ratios, provisioning adequacy, and credit quality.'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="pillar-card">
            <b style="color:#9333ea;">3. Productivity (P)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'كفاءة تشغيل الأصول وتوظيف الموارد البشرية والتقنية.' if lang=='ar' else 'Asset utilization efficiency and workforce productivity metrics.'}
            </span>
        </div>
        <div class="pillar-card">
            <b style="color:#059669;">4. Profitability (P)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'العائد على الأصول، هامش الفائدة الصافي، واستدامة الأرباح.' if lang=='ar' else 'Net Interest Margin, ROA sustainability, and revenue quality.'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""
        <div class="pillar-card">
            <b style="color:#d97706;">5. Efficiency (E)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'ضبط نسبة التكلفة إلى الدخل وترشيد المصاريف التشغيلية.' if lang=='ar' else 'Cost-to-income rationalization and overhead burden management.'}
            </span>
        </div>
        <div class="pillar-card">
            <b style="color:#0891b2;">6. Liquidity (L) & 7. Openness (O)</b><br>
            <span style="font-size:12px; color:#64748b;">
                {'الأصول عالية السيولة، استحقاق الالتزامات، والتحوط لمخاطر العملات والفوائد.' if lang=='ar' else 'HQLA liquidity, maturity mismatch, and FX/Interest rate market sensitivity.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ================= SECTION 2: MACRO SECTOR ASSESSMENT =================
st.subheader(T['sec_sector'])
comp_df = get_sector_data(df_data, selected_year, lang=lang)

k1, k2, k3, k4 = st.columns(4)
total_b = len(comp_df)
high_n = len(comp_df[comp_df['Supervisory Status'].str.contains('High|عالية', na=False)])
watch_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Watch|Early|مراقبة|إنذار', na=False)])
stable_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Stable|مستقر', na=False)])

with k1:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["monitored_banks"]}</div><div style="font-size:22px; font-weight:800; color:#0f172a;">{total_b} Banks</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["high_risk"]}</div><div style="font-size:22px; font-weight:800; color:#ef4444;">{high_n}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["watch_warning"]}</div><div style="font-size:22px; font-weight:800; color:#f59e0b;">{watch_n}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["stable"]}</div><div style="font-size:22px; font-weight:800; color:#10b981;">{stable_n}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_chart, col_table = st.columns([5, 7])
with col_chart:
    fig_bar = px.bar(
        comp_df,
        x='Bank',
        y='Risk Score',
        color='Risk Score',
        color_continuous_scale='RdYlGn_r',
        text='Risk Score',
        labels={'Risk Score': T['risk_score'], 'Bank': T['col_bank']}
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Early Warning Threshold (50.00)")
    fig_bar.update_layout(height=340, margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_table:
    display_comp_df = comp_df.rename(columns={
        'Bank': T['col_bank'],
        'Risk Score': T['col_risk'],
        '90D Stress Prob (%)': T['col_prob'],
        'Supervisory Status': T['col_status'],
        'Capital': T['col_cap'],
        'Asset Quality': T['col_ast'],
        'Productivity': T['col_prd'],
        'Profitability': T['col_prf'],
        'Efficiency': T['col_eff'],
        'Liquidity': T['col_liq'],
        'Openness': T['col_opn']
    })
    st.dataframe(
        display_comp_df.style.format({
            T['col_risk']: '{:.2f}',
            T['col_prob']: '{:.2f}%',
            T['col_cap']: '{:.1f}',
            T['col_ast']: '{:.1f}',
            T['col_prd']: '{:.1f}',
            T['col_prf']: '{:.1f}',
            T['col_eff']: '{:.1f}',
            T['col_liq']: '{:.1f}',
            T['col_opn']: '{:.1f}'
        }),
        use_container_width=True,
        height=340
    )

st.markdown("---")

# ================= SECTION 3: INSTITUTIONAL DEEP-DIVE =================
st.subheader(f"{T['sec_deepdive']} — {selected_bank} ({selected_year})")

analysis, insights, var_df = get_analysis(df_data, selected_bank, selected_year)
assessment = analysis['Assessment']
cat_scores = analysis['Category_Scores']
raw_status = assessment['Status']
st_text = STATUS_MAP.get(raw_status, {}).get(lang, raw_status)

top_esc_cat = insights['Top_Risk_Escalators'][0]['Category'] if insights['Top_Risk_Escalators'] else 'Capital'
adv_text = RECOMMENDATIONS.get(top_esc_cat, RECOMMENDATIONS['Capital'])[lang]

c_d1, c_d2, c_d3, c_d4 = st.columns(4)
with c_d1:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{selected_bank}</div><div style="font-size:22px; font-weight:800; color:#0f172a;">{selected_year}</div></div>', unsafe_allow_html=True)
with c_d2:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["risk_score"]}</div><div style="font-size:22px; font-weight:800; color:#0f172a;">{assessment["Risk_Score"]} / 100</div></div>', unsafe_allow_html=True)
with c_d3:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["stress_prob"]}</div><div style="font-size:22px; font-weight:800; color:#0f172a;">{assessment["Stress_Probability_90D"]}%</div></div>', unsafe_allow_html=True)
with c_d4:
    st.markdown(f'<div class="stat-box"><div style="font-size:11px; font-weight:700; color:#64748b;">{T["supervisory_status"]}</div><div style="font-size:20px; font-weight:800; color:{assessment["Color"]};">{st_text}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_radar, col_gauge = st.columns([6, 4])
with col_radar:
    st.markdown(f"##### {T['radar_title']}")
    cat_labels = [
        f"Capital ({cat_scores['Capital']:.1f})",
        f"Asset Q. ({cat_scores['Asset Quality']:.1f})",
        f"Productivity ({cat_scores['Productivity']:.1f})",
        f"Profitability ({cat_scores['Profitability']:.1f})",
        f"Efficiency ({cat_scores['Efficiency']:.1f})",
        f"Liquidity ({cat_scores['Liquidity']:.1f})",
        f"Openness ({cat_scores['Openness']:.1f})"
    ]
    values = [cat_scores['Capital'], cat_scores['Asset Quality'], cat_scores['Productivity'], cat_scores['Profitability'], cat_scores['Efficiency'], cat_scores['Liquidity'], cat_scores['Openness']]
    norm_values = [40, 35, 30, 35, 30, 35, 30]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=cat_labels + [cat_labels[0]], fill='toself', name=selected_bank, line_color='#2563eb', fillcolor='rgba(37, 99, 235, 0.25)'))
    fig_radar.add_trace(go.Scatterpolar(r=norm_values + [norm_values[0]], theta=cat_labels + [cat_labels[0]], fill='toself', name='Sector Benchmark', line_color='#94a3b8', line_dash='dash', fillcolor='rgba(148, 163, 184, 0.15)'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_radar, use_container_width=True)

with col_gauge:
    st.markdown(f"##### {T['gauge_title']}")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=assessment['Risk_Score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{selected_bank} Composite Index", 'font': {'size': 13}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': assessment['Color']},
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {'line': {'color': "#ef4444", 'width': 3}, 'thickness': 0.75, 'value': 50.0}
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown(f"##### {T['xai_title']}")
col_esc, col_mit = st.columns(2)
with col_esc:
    st.markdown(f"**{T['escalators']}**")
    if insights['Top_Risk_Escalators']:
        for esc in insights['Top_Risk_Escalators']:
            st.warning(f"**{esc['Category']}:** Sub-Score = `{esc['Sub_Score']:.2f}` (+`{esc['Impact_Score']:.2f}` pts)")
    else:
        st.success("No severe risk escalators detected.")
with col_mit:
    st.markdown(f"**{T['mitigators']}**")
    if insights['Top_Risk_Mitigators']:
        for mit in insights['Top_Risk_Mitigators']:
            st.success(f"**{mit['Category']}:** Sub-Score = `{mit['Sub_Score']:.2f}`")

st.info(f"**{T['advisory_title']}** {adv_text}")

st.markdown("---")

# ================= SECTION 4: DETAILED VARIANCE MATRIX =================
st.subheader(T['sec_variance'])
search_q = st.text_input(T['search'], "")

clean_var_df = var_df.copy()
clean_var_df['Indicator'] = clean_var_df['Indicator'].apply(lambda x: clean_indicator_string(x, lang=lang))

if search_q:
    clean_var_df = clean_var_df[clean_var_df['Indicator'].str.contains(search_q, case=False, na=False)]

clean_var_df = clean_var_df.rename(columns={
    'Indicator': 'Indicator / KPI Name' if lang == 'en' else 'اسم المؤشر المالي',
    'Bank_Value': f'{selected_bank} Value' if lang == 'en' else f'قيمة {selected_bank}',
    'Industry_Norm': 'Sector Benchmark' if lang == 'en' else 'معيار القطاع',
    'Absolute_Diff': 'Variance (Abs)' if lang == 'en' else 'الانحراف المطلق',
    'Variance_Pct': 'Variance (%)' if lang == 'en' else 'نسبة التغير (%)'
})

val_col = f'{selected_bank} Value' if lang == 'en' else f'قيمة {selected_bank}'
norm_col = 'Sector Benchmark' if lang == 'en' else 'معيار القطاع'
diff_col = 'Variance (Abs)' if lang == 'en' else 'الانحراف المطلق'
pct_col = 'Variance (%)' if lang == 'en' else 'نسبة التغير (%)'

st.dataframe(
    clean_var_df.style.format({
        val_col: '{:,.2f}',
        norm_col: '{:,.2f}',
        diff_col: '{:,.2f}',
        pct_col: '{:+.2f}%'
    }),
    use_container_width=True,
    height=360
)

# ================= ACADEMIC FOOTER =================
st.markdown("""
<div class="footer-box">
    <b>Early Warning Model for Banks' Risk (EWM-BR)</b> — CAPPELO Empirical Framework<br>
    Academic & Supervisory Intelligence Platform © 2026. All Rights Reserved.
</div>
""", unsafe_allow_html=True)