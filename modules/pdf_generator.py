import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_bank_risk_pdf(bank_name, year, assessment, cat_scores, insights, advisory):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceAfter=15
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    story = []

    story.append(Paragraph("EWM-BR Bank Risk Assessment Report", title_style))
    story.append(Paragraph(f"Comprehensive Financial Soundness Evaluation | Bank: {bank_name} ({year})", subtitle_style))
    story.append(Spacer(1, 8))

    summary_data = [
        ["Target Bank", bank_name, "Financial Year", str(year)],
        ["Overall Risk Score", f"{assessment['Risk_Score']} / 100", "90-Day Stress Prob.", f"{assessment['Stress_Probability_90D']}%"],
        ["Health Classification", assessment['Status'], "Supervisory Action", assessment['Action']]
    ]

    t_summary = Table(summary_data, colWidths=[120, 150, 120, 150])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    story.append(Paragraph("CAPPELO Pillar Sub-Scores (Normalized 0 - 100)", section_style))
    
    cat_rows = [["Pillar Dimension", "Assigned Weight", "Risk Sub-Score", "Weighted Points"]]
    weights = {
        'Capital': 0.20,
        'Asset Quality': 0.25,
        'Productivity': 0.10,
        'Profitability': 0.15,
        'Efficiency': 0.10,
        'Liquidity': 0.15,
        'Openness': 0.05
    }
    
    for cat, score in cat_scores.items():
        w = weights.get(cat, 0.10)
        cat_rows.append([cat, f"{int(w*100)}%", f"{score:.2f}", f"{score*w:.2f}"])

    t_cats = Table(cat_rows, colWidths=[150, 110, 140, 140])
    t_cats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cats)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Explainable AI (XAI) Risk Drivers", section_style))
    
    escalators = insights.get('Top_Risk_Escalators', [])
    esc_text = "<b>Key Risk Escalators (Increasing Risk):</b><br/>"
    if escalators:
        for esc in escalators:
            esc_text += f"• Pillar {esc['Category']}: Sub-Score = {esc['Sub_Score']:.2f} (Added {esc['Impact_Score']:.2f} risk points)<br/>"
    else:
        esc_text += "• No high-risk escalators detected.<br/>"
    story.append(Paragraph(esc_text, body_style))
    story.append(Spacer(1, 6))

    mitigators = insights.get('Top_Risk_Mitigators', [])
    mit_text = "<b>Key Risk Mitigators (Safety Buffers):</b><br/>"
    if mitigators:
        for mit in mitigators:
            mit_text += f"• Pillar {mit['Category']}: Sub-Score = {mit['Sub_Score']:.2f} (Reduced exposure significantly)<br/>"
    story.append(Paragraph(mit_text, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Supervisory Advisory", section_style))
    story.append(Paragraph(f"<b>Recommendation:</b> {advisory}", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    test_assessment = {
        'Risk_Score': 49.68,
        'Stress_Probability_90D': 49.68,
        'Status': 'Watch (مراقبة)',
        'Action': 'Monitor deteriorating KPIs'
    }
    test_cat_scores = {
        'Capital': 62.10,
        'Asset Quality': 63.89,
        'Productivity': 20.00,
        'Profitability': 49.40,
        'Efficiency': 68.76,
        'Liquidity': 0.00,
        'Openness': 100.00
    }
    test_insights = {
        'Top_Risk_Escalators': [
            {'Category': 'Asset Quality', 'Sub_Score': 63.89, 'Impact_Score': 15.97},
            {'Category': 'Capital', 'Sub_Score': 62.10, 'Impact_Score': 12.42}
        ],
        'Top_Risk_Mitigators': [
            {'Category': 'Liquidity', 'Sub_Score': 0.00, 'Impact_Score': 0.00}
        ]
    }
    test_advisory = "Tighten credit policies and reinforce NPL provisioning buffers."

    pdf_bytes = create_bank_risk_pdf(
        "Qatar Al Watani Bank 1964",
        2024,
        test_assessment,
        test_cat_scores,
        test_insights,
        test_advisory
    )

    with open("sample_test_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    print("✅ تم توليد ملف PDF التجريبي بنجاح.")