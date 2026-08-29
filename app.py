import os
import sys
import re
import streamlit as st
import streamlit.components.v1 as components
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
    page_title="EWM-BR | Banking Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Tajawal:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Tajawal', sans-serif;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease-in-out;
}
.kpi-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
}
.kpi-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
}
.kpi-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

I18N = {
    'en': {
        'title': "🏦 Early Warning System for Banking Risk (EWM-BR)",
        'subtitle': "Predictive Financial Soundness Evaluation based on CAPPELO Framework (Groups 1–7)",
        'nav_landing': "🏠 Platform Overview & CAPPELO Architecture",
        'nav_sector': "🏢 Cross-Sector Master View",
        'nav_single': "🔍 Single Institution Assessment",
        'total_banks': "Monitored Institutions",
        'high_risk': "High Risk Exposure",
        'watch_warning': "Watch / Early Warning",
        'stable': "Stable Soundness",
        'sec_ranking': "📊 Banking Sector Composite Risk Distribution",
        'sec_matrix': "📋 Cross-Institutional Soundness Matrix (All Monitored Banks)",
        'target_bank': "Target Institution",
        'risk_score': "Overall Risk Score",
        'stress_prob': "90-Day Stress Probability",
        'supervisory_status': "Supervisory Classification",
        'radar_title': "🎯 CAPPELO Multidimensional Risk Profile",
        'gauge_title': "⏱️ Composite Risk Index Gauge",
        'xai_title': "🔍 Explainable AI (XAI) Risk Decomposition",
        'escalators': "🚨 Key Risk Escalators (Increasing Vulnerability)",
        'mitigators': "🛡️ Key Risk Mitigators (Safety Buffers)",
        'advisory_title': "💡 Supervisory Action Directive:",
        'variance_title': "📋 Core Indicators Variance Matrix vs. Sector Benchmark",
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
        'title': "🏦 منصة الإنذار المبكر للمخاطر المصرفية (EWM-BR)",
        'subtitle': "التقييم والاستشراف الذكي لسلامة البنوك استناداً لمحاور CAPPELO السبعة (Groups 1–7)",
        'nav_landing': "🏠 التعريف بالمنصة ومحاور CAPPELO",
        'nav_sector': "🏢 الشاشة الشاملة للقطاع المصرفي",
        'nav_single': "🔍 التحليل التفصيلي لمصرف فردي",
        'total_banks': "المصارف الخاضعة للرقابة",
        'high_risk': "مؤسسات عالية المخاطر",
        'watch_warning': "مؤسسات قيد المراقبة / إنذار",
        'stable': "مؤسسات مستقرة",
        'sec_ranking': "📊 توزيع وترتيب مؤشرات المخاطر للقطاع المصرفي",
        'sec_matrix': "📋 مصفوفة السلامة المالية المقارنة لكافة البنوك",
        'target_bank': "المصرف الخاضع للتقييم",
        'risk_score': "درجة المخاطر المركبة",
        'stress_prob': "احتمالية الضغط المالي (90 يوماً)",
        'supervisory_status': "التصنيف الرقابي",
        'radar_title': "🎯 رادار أبعاد السلامة المصرفية (CAPPELO)",
        'gauge_title': "⏱️ عداد مؤشر المخاطر التراكمي",
        'xai_title': "🔍 التفسير السببي وتفكيك محركات الخطر (XAI)",
        'escalators': "🚨 المحاور الأكثر رفعاً للمخاطر",
        'mitigators': "🛡️ محركات الأمان وتخفيف المخاطر",
        'advisory_title': "💡 التوجيه والتوصية الرقابية:",
        'variance_title': "📋 مصفوفة انحراف المؤشرات التفصيلية عن معيار الصناعة",
        'search': "بحث في المؤشرات المالية:",
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

# Sidebar Navigation
with st.sidebar:
    st.markdown("### ⚙️ Platform Navigation")
    enable_arabic = st.toggle("🌐 إظهار الواجهة باللغة العربية (Arabic)", value=False)
    lang = 'ar' if enable_arabic else 'en'
    T = I18N[lang]

    st.markdown("---")
    all_years = sorted(df_data['Year'].unique().tolist(), reverse=True)
    selected_year = st.selectbox("Fiscal Year | السنة المالية:", all_years)

    nav_mode = st.radio(
        "Navigation Mode | نمط العرض:",
        [T['nav_landing'], T['nav_sector'], T['nav_single']],
        index=0
    )

    bank_list = sorted(df_data[df_data['Year'] == selected_year]['Bank'].unique().tolist())
    if nav_mode == T['nav_single']:
        selected_bank = st.selectbox("Select Target Bank | المصرف الخاضع للتقييم:", bank_list)
    else:
        selected_bank = bank_list[0]

comp_df = get_sector_data(df_data, selected_year, lang=lang)

# ----------------- VIEW 1: EMBEDDED TAILWIND LANDING PAGE -----------------
if nav_mode == T['nav_landing']:
    html_content = f"""
    <!DOCTYPE html>
    <html lang="{'ar' if lang == 'ar' else 'en'}" dir="{'rtl' if lang == 'ar' else 'ltr'}">
    <head>
        <meta charset="UTF-8">
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            body {{ font-family: {'"Tajawal"' if lang == 'ar' else '"Plus Jakarta Sans"'}, sans-serif; }}
            .glass-card {{
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .typewriter-cursor {{
                display: inline-block;
                width: 3px;
                background-color: #38bdf8;
                animation: blink 0.8s infinite;
            }}
            @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
        </style>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-4 sm:p-8">
        <div class="max-w-6xl mx-auto">
            <!-- Header Badge -->
            <div class="text-center pt-4 pb-6">
                <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card border border-blue-500/30 text-sky-400 text-xs sm:text-sm font-semibold mb-6">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>CAPPELO Framework Core Empirical Analysis (2023–2024)</span>
                </div>
                
                <h1 class="text-3xl sm:text-5xl font-extrabold text-white tracking-tight mb-4 leading-tight">
                    {'منظومة الاستشراف والإنذار المبكر للمخاطر المصرفية' if lang == 'ar' else 'Early Warning Model for Banks\' Risk (EWM-BR)'}<br>
                    <span class="bg-gradient-to-r from-sky-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        {'تقييم السلامة المصرفية بإطار CAPPELO' if lang == 'ar' else 'CAPPELO Supervisory Intelligence Framework'}
                    </span>
                </h1>
                
                <div class="min-h-[60px] text-base sm:text-lg text-slate-300 max-w-3xl mx-auto mb-8 font-medium">
                    <span id="typewriter"></span><span class="typewriter-cursor">&nbsp;</span>
                </div>
            </div>

            <!-- Key Metric Numbers -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
                <div class="glass-card rounded-2xl p-5 text-center border border-slate-800">
                    <div class="text-3xl font-extrabold text-sky-400 mb-1">8</div>
                    <div class="text-xs text-slate-400 font-semibold">{'بنوك مجهلة تحت الرقابة' if lang == 'ar' else 'Monitored Anonymized Banks'}</div>
                </div>
                <div class="glass-card rounded-2xl p-5 text-center border border-slate-800">
                    <div class="text-3xl font-extrabold text-cyan-400 mb-1">7</div>
                    <div class="text-xs text-slate-400 font-semibold">{'محاور CAPPELO التحليلية' if lang == 'ar' else 'CAPPELO Core Dimensions'}</div>
                </div>
                <div class="glass-card rounded-2xl p-5 text-center border border-slate-800">
                    <div class="text-3xl font-extrabold text-emerald-400 mb-1">144</div>
                    <div class="text-xs text-slate-400 font-semibold">{'مؤشراً مالياً معيارياً' if lang == 'ar' else 'Standardized Soundness KPIs'}</div>
                </div>
                <div class="glass-card rounded-2xl p-5 text-center border border-slate-800">
                    <div class="text-3xl font-extrabold text-indigo-400 mb-1">90 {'يوم' if lang == 'ar' else 'Days'}</div>
                    <div class="text-xs text-slate-400 font-semibold">{'أفق استشراف الضغط المالي' if lang == 'ar' else 'Stress Horizon Forecast'}</div>
                </div>
            </div>

            <!-- 7 Pillars Grid -->
            <div class="text-center mb-8">
                <h2 class="text-2xl font-bold text-white mb-2">{'محاور التقييم السبعة لإطار CAPPELO' if lang == 'ar' else 'The 7 CAPPELO Analytical Pillars'}</h2>
                <p class="text-slate-400 text-xs sm:text-sm">{'نمذجة رياضية متكاملة لتقييم الملاءة المالية وحساسية الأسواق والسيولة' if lang == 'ar' else 'Quantitative econometric framework mapping financial resilience across core banking risk dimensions'}</p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pb-8">
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-blue-400 font-bold mb-2">1. Capital Adequacy (C)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'كفاية رأس المال والرافعة المالية ومطابقة بازل 3 لامتصاص الصدمات.' if lang == 'ar' else 'Capital buffer, leverage limits, and risk-weighted asset capacity.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-red-400 font-bold mb-2">2. Asset Quality (A)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'جودة المحفظة الائتمانية ورصد القروض غير المنتظمة (NPL) والمخصصات.' if lang == 'ar' else 'Non-performing loan ratios, provisioning adequacy, and credit quality.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-purple-400 font-bold mb-2">3. Productivity (P)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'كفاءة تشغيل الأصول وتوظيف الموارد البشرية والتقنية.' if lang == 'ar' else 'Asset utilization efficiency and workforce productivity metrics.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-emerald-400 font-bold mb-2">4. Profitability (P)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'العائد على الأصول (ROA)، هامش الفائدة الصافي، واستدامة الأرباح.' if lang == 'ar' else 'Net Interest Margin (NIM), ROA sustainability, and revenue quality.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-amber-400 font-bold mb-2">5. Efficiency (E)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'ضبط نسبة التكلفة إلى الدخل وترشيد المصاريف التشغيلية.' if lang == 'ar' else 'Cost-to-income rationalization and overhead burden management.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800">
                    <div class="text-cyan-400 font-bold mb-2">6. Liquidity (L)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'الأصول عالية السيولة واستقرار هيكل التمويل واستحقاق الالتزامات.' if lang == 'ar' else 'High-Quality Liquid Assets (HQLA) and short-term maturity profiles.'}</p>
                </div>
                <div class="glass-card rounded-xl p-5 border border-slate-800 md:col-span-2 lg:col-span-3">
                    <div class="text-indigo-400 font-bold mb-2">7. Openness & Sensitivity to Market Risk (O)</div>
                    <p class="text-slate-400 text-xs leading-relaxed">{'مراقبة تقلبات أسعار الفائدة والعملات الأجنبية وأدوات التحوط.' if lang == 'ar' else 'Foreign exchange structural exposure and market risk sensitivity gaps.'}</p>
                </div>
            </div>
        </div>

        <script>
            lucide.createIcons();
            const sentences = {'["منظومة رقابية متقدمة ترصد الاستقرار المالي عبر إطار CAPPELO المعتمد.", "استخراج فوري للمؤشرات المعيارية لـ 8 بنوك مجهلة لتأكيد الحياد والسرية.", "توليد تقارير PDF تنفيذية وتفسير سببي متقدم (XAI) للمخاطر."]' if lang == 'ar' else '["Empirical early warning system assessing banking vulnerability via CAPPELO framework.", "Anonymized 8-bank panel dataset ensuring regulatory objectivity and rigor.", "Automated executive PDF reporting with Explainable AI risk decomposition."]'};
            let sIdx = 0, cIdx = 0, isDel = false;
            const el = document.getElementById("typewriter");
            function type() {{
                const cur = sentences[sIdx];
                el.textContent = isDel ? cur.substring(0, cIdx - 1) : cur.substring(0, cIdx + 1);
                cIdx = isDel ? cIdx - 1 : cIdx + 1;
                let t = isDel ? 20 : 35;
                if (!isDel && cIdx === cur.length) {{ isDel = true; t = 2200; }}
                else if (isDel && cIdx === 0) {{ isDel = false; sIdx = (sIdx + 1) % sentences.length; t = 400; }}
                setTimeout(type, t);
            }}
            document.addEventListener("DOMContentLoaded", type);
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=880, scrolling=True)

# ----------------- VIEW 2: CROSS-SECTOR MASTER DASHBOARD -----------------
elif nav_mode == T['nav_sector']:
    col_head, col_btn = st.columns([4, 2])
    with col_head:
        st.title(T['title'])
        st.caption(f"{T['subtitle']} | Fiscal Year: {selected_year}")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        comp_df_en = get_sector_data(df_data, selected_year, lang='en')
        pdf_bytes = create_sector_risk_pdf(comp_df_en, selected_year)
        st.download_button(
            label=T['download_sector'],
            data=pdf_bytes,
            file_name=f"EWM_BR_Sector_Report_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    total_b = len(comp_df)
    high_n = len(comp_df[comp_df['Supervisory Status'].str.contains('High|عالية', na=False)])
    watch_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Watch|Early|مراقبة|إنذار', na=False)])
    stable_n = len(comp_df[comp_df['Supervisory Status'].str.contains('Stable|مستقر', na=False)])

    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["total_banks"]}</div><div class="kpi-value">{total_b} Banks</div><div class="kpi-sub">Coverage: 100%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["high_risk"]}</div><div class="kpi-value" style="color:#ef4444;">{high_n}</div><div class="kpi-sub">Critical Alert</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["watch_warning"]}</div><div class="kpi-value" style="color:#f59e0b;">{watch_n}</div><div class="kpi-sub">Heightened Vigilance</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["stable"]}</div><div class="kpi-value" style="color:#10b981;">{stable_n}</div><div class="kpi-sub">Within Safety Norm</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(T['sec_ranking'])
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
    fig_bar.update_layout(height=380, margin=dict(t=25, b=25))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader(T['sec_matrix'])
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
        height=320
    )

# ----------------- VIEW 3: SINGLE BANK DEEP-DIVE -----------------
else:
    analysis, insights, var_df = get_analysis(df_data, selected_bank, selected_year)
    assessment = analysis['Assessment']
    cat_scores = analysis['Category_Scores']
    raw_status = assessment['Status']
    st_text = STATUS_MAP.get(raw_status, {}).get(lang, raw_status)

    top_esc_cat = insights['Top_Risk_Escalators'][0]['Category'] if insights['Top_Risk_Escalators'] else 'Capital'
    adv_text = RECOMMENDATIONS.get(top_esc_cat, RECOMMENDATIONS['Capital'])[lang]
    adv_en = RECOMMENDATIONS.get(top_esc_cat, RECOMMENDATIONS['Capital'])['en']

    col_head, col_btn = st.columns([4, 2])
    with col_head:
        st.title(f"🏦 {selected_bank} — {selected_year}")
        st.caption(f"{T['subtitle']} | Target: {selected_bank}")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_s_bytes = create_bank_risk_pdf(
            bank_name=selected_bank,
            year=selected_year,
            assessment=analysis['Assessment'],
            cat_scores=analysis['Category_Scores'],
            insights=insights,
            advisory=adv_en
        )
        st.download_button(
            label=T['download_single'],
            data=pdf_s_bytes,
            file_name=f"EWM_BR_{selected_bank}_{selected_year}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["target_bank"]}</div><div class="kpi-value" style="font-size:20px;">{selected_bank}</div><div class="kpi-sub">Year: {selected_year}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["risk_score"]}</div><div class="kpi-value">{assessment["Risk_Score"]} / 100</div><div class="kpi-sub">Weighted Composite</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["stress_prob"]}</div><div class="kpi-value">{assessment["Stress_Probability_90D"]}%</div><div class="kpi-sub">Predictive Horizon</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">{T["supervisory_status"]}</div><div class="kpi-value" style="color:{assessment["Color"]}; font-size:19px;">{st_text}</div><div class="kpi-sub">Official Action Rating</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_radar, col_gauge = st.columns([6, 4])
    with col_radar:
        st.subheader(T['radar_title'])
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
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_gauge:
        st.subheader(T['gauge_title'])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment['Risk_Score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Risk Gauge (0-100)", 'font': {'size': 13}},
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
        fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader(T['xai_title'])
    col_esc, col_mit = st.columns(2)
    with col_esc:
        st.markdown(f"##### {T['escalators']}")
        if insights['Top_Risk_Escalators']:
            for esc in insights['Top_Risk_Escalators']:
                st.warning(f"**{esc['Category']}:** Sub-Score = `{esc['Sub_Score']:.2f}` (+`{esc['Impact_Score']:.2f}` pts)")
        else:
            st.success("No severe risk escalators detected.")
    with col_mit:
        st.markdown(f"##### {T['mitigators']}")
        if insights['Top_Risk_Mitigators']:
            for mit in insights['Top_Risk_Mitigators']:
                st.success(f"**{mit['Category']}:** Sub-Score = `{mit['Sub_Score']:.2f}`")

    st.info(f"**{T['advisory_title']}** {adv_text}")

    st.markdown("---")
    st.subheader(T['variance_title'])
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
        height=320
    )