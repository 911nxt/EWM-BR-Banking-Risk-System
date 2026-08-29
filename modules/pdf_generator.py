import io
import re
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def clean_status_text(val):
    text = str(val)
    if "High" in text:
        return "High Risk"
    elif "Early" in text:
        return "Early Warning"
    elif "Watch" in text:
        return "Watch"
    elif "Stable" in text:
        return "Stable"
    return re.sub(r'[\u0600-\u06FF]', '', text).replace('()', '').strip()

def create_sector_risk_pdf(comp_df, year):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceAfter=12
    )

    story = []
    story.append(Paragraph("EWM-BR Comprehensive Banking Sector Risk Report", title_style))
    story.append(Paragraph(f"Cross-Sectional Financial Soundness Evaluation (CAPPELO Framework) | Fiscal Year: {year}", subtitle_style))

    headers = ['Bank', 'Risk Score', '90D Stress', 'Status', 'Capital', 'Asset Q.', 'Productivity', 'Profitability', 'Efficiency', 'Liquidity', 'Openness']
    table_data = [headers]

    for _, row in comp_df.iterrows():
        status_clean = clean_status_text(row['Supervisory Status'])
        table_data.append([
            str(row['Bank']),
            f"{row['Risk Score']:.2f}",
            f"{row['90D Stress Prob (%)']:.2f}%",
            status_clean,
            f"{row['Capital']:.1f}",
            f"{row['Asset Quality']:.1f}",
            f"{row['Productivity']:.1f}",
            f"{row['Profitability']:.1f}",
            f"{row['Efficiency']:.1f}",
            f"{row['Liquidity']:.1f}",
            f"{row['Openness']:.1f}"
        ])

    t = Table(table_data, colWidths=[65, 65, 75, 85, 55, 55, 65, 65, 55, 55, 55])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#0f172a'), alignment=1, spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor('#64748b'), alignment=1, spaceAfter=12)
    section_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#1e293b'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('BodyDark', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#334155'))

    story = []
    story.append(Paragraph("EWM-BR Bank Risk Assessment Report", title_style))
    story.append(Paragraph(f"Financial Soundness Evaluation | Bank: {bank_name} ({year})", subtitle_style))

    status_clean = clean_status_text(assessment['Status'])
    summary_data = [
        ["Target Bank", bank_name, "Financial Year", str(year)],
        ["Overall Risk Score", f"{assessment['Risk_Score']} / 100", "90-Day Stress Prob.", f"{assessment['Stress_Probability_90D']}%"],
        ["Health Classification", status_clean, "Supervisory Action", clean_status_text(assessment['Action'])]
    ]
    t_sum = Table(summary_data, colWidths=[120, 150, 120, 150])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sum)

    story.append(Paragraph("CAPPELO Pillar Sub-Scores (Normalized 0 - 100)", section_style))
    cat_rows = [["Pillar Dimension", "Assigned Weight", "Risk Sub-Score", "Weighted Points"]]
    weights = {'Capital': 0.20, 'Asset Quality': 0.25, 'Productivity': 0.10, 'Profitability': 0.15, 'Efficiency': 0.10, 'Liquidity': 0.15, 'Openness': 0.05}
    for cat, score in cat_scores.items():
        w = weights.get(cat, 0.10)
        cat_rows.append([cat, f"{int(w*100)}%", f"{score:.2f}", f"{score*w:.2f}"])

    t_cats = Table(cat_rows, colWidths=[150, 110, 140, 140])
    t_cats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_cats)

    story.append(Paragraph("Supervisory Advisory", section_style))
    story.append(Paragraph(f"<b>Recommendation:</b> {advisory}", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()