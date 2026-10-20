"""Downloadable HTML report."""
import html
from io import BytesIO

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart


INK = colors.HexColor("#172033")
BLUE = colors.HexColor("#3157A4")
MUTED = colors.HexColor("#5B6475")
PALE_BLUE = colors.HexColor("#EAF0FC")
PALE_GRAY = colors.HexColor("#F3F5F8")


def _pdf_text(value) -> str:
    """Escape user-provided text and replace glyphs unsupported by built-in PDF fonts."""
    text = str(value).encode("latin-1", "replace").decode("latin-1")
    return html.escape(text).replace("\n", "<br/>")


class _Footer(Flowable):
    def __init__(self, title):
        super().__init__()
        self.title = title

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        width, _ = letter
        canvas.setStrokeColor(colors.HexColor("#D8DEE9"))
        canvas.line(0, 0.38 * inch, width - 1.2 * inch, 0.38 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0, 0.18 * inch, self.title[:100])
        canvas.drawRightString(width - 1.2 * inch, 0.18 * inch, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()
        self.width = width - 1.2 * inch
        self.height = 0.45 * inch


def _chart(labels: list[str], values: list[float], maximum: float):
    drawing = Drawing(450, 190)
    chart = VerticalBarChart()
    chart.x = 48
    chart.y = 32
    chart.height = 125
    chart.width = 380
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.angle = 25
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.dx = -5
    chart.categoryAxis.labels.dy = -15
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = maximum if maximum > 0 else 1
    chart.valueAxis.valueStep = max(maximum / 4, 1)
    chart.bars[0].fillColor = BLUE
    drawing.add(chart)
    return drawing


def to_pdf(
    name,
    profile,
    quality,
    findings,
    df=None,
    cleaning_log=None,
    analysis_tables=None,
    analysis_methods=None,
) -> bytes:
    """Build an evidence-focused, printable PDF report."""
    buffer = BytesIO()
    page_width, _ = letter
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title=f"FlyBrain AutoAnalyst - {name}",
        author="FlyBrain AutoAnalyst",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        alignment=TA_CENTER,
        textColor=INK,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        textColor=MUTED,
        fontSize=9,
        leading=13,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="ReportSection",
        parent=styles["Heading2"],
        textColor=BLUE,
        spaceBefore=15,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SmallCell",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    ))
    styles.add(ParagraphStyle(
        name="FindingTitle",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        textColor=INK,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="FindingBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=4,
    ))
    story = [
        Paragraph("FlyBrain AutoAnalyst", styles["ReportTitle"]),
        Paragraph(f"Analysis report for {_pdf_text(name)}", styles["ReportSubtitle"]),
        Paragraph("Executive summary", styles["ReportSection"]),
        Paragraph(
            f"The analyzed dataset contains <b>{profile['n_rows']:,} rows</b> and "
            f"<b>{profile['n_cols']:,} columns</b>. "
            f"Overall missing-cell rate: <b>{profile['missing_rate']:.1%}</b>. "
            f"Quality issues: <b>{len(quality)}</b>; statistically supported findings: "
            f"<b>{len(findings)}</b>.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
    ]
    overview_data = [
        ["Dataset metric", "Value"],
        ["Rows", f"{profile['n_rows']:,}"],
        ["Columns", f"{profile['n_cols']:,}"],
        ["Missing-cell rate", f"{profile['missing_rate']:.2%}"],
        ["Duplicate-row rate", f"{profile['duplicate_rate']:.2%}"],
        ["Numeric columns", str(len(profile.get("numeric_columns", [])))],
        ["Categorical columns", str(len(profile.get("categorical_columns", [])))],
        ["Datetime columns", str(len(profile.get("datetime_columns", [])))],
    ]
    overview = Table(overview_data, colWidths=[2.5 * inch, 3.8 * inch], repeatRows=1)
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([overview, Spacer(1, 8)])

    if analysis_methods:
        story.extend([
            Paragraph("Analysis methods", styles["ReportSection"]),
            Paragraph(
                _pdf_text(", ".join(analysis_methods)),
                styles["BodyText"],
            ),
        ])

    if df is not None and not df.empty:
        missing = df.isna().mean().sort_values(ascending=False)
        missing = missing[missing > 0].head(10)
        if not missing.empty:
            labels = [str(label)[:16] for label in missing.index]
            values = (missing * 100).tolist()
            story.extend([
                Paragraph("Missingness", styles["ReportSection"]),
                Paragraph("Missing values by column (%)", styles["Heading3"]),
                _chart(labels, values, 100),
                Spacer(1, 8),
            ])
        numeric = df.select_dtypes(include="number").columns.tolist()
        if numeric:
            column = numeric[0]
            values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()
            if len(values):
                counts, edges = np.histogram(values, bins=min(10, max(3, int(np.sqrt(len(values))))))
                labels = [
                    f"{edges[i]:.3g}-{edges[i + 1]:.3g}"
                    for i in range(len(counts))
                ]
                story.extend([
                    Paragraph("Distribution snapshot", styles["ReportSection"]),
                    Paragraph(f"Histogram: {_pdf_text(column)}", styles["Heading3"]),
                    _chart(labels, counts.tolist(), float(max(counts))),
                    Spacer(1, 8),
                ])

    story.append(Paragraph("Data quality", styles["ReportSection"]))
    if quality:
        quality_rows = [["Severity", "Issue", "Details"]]
        quality_rows.extend([
            [
                Paragraph(_pdf_text(item.get("severity", "info")).upper(), styles["SmallCell"]),
                Paragraph(_pdf_text(item.get("issue", "quality check")).replace("_", " "), styles["SmallCell"]),
                Paragraph(_pdf_text(item.get("detail", "")), styles["SmallCell"]),
            ]
            for item in quality
        ])
        quality_table = Table(
            quality_rows,
            colWidths=[0.9 * inch, 1.45 * inch, 3.95 * inch],
            repeatRows=1,
        )
        quality_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(quality_table)
    else:
        story.append(Paragraph("No data-quality issues were flagged.", styles["BodyText"]))

    if cleaning_log:
        story.append(Paragraph("Cleaning audit trail", styles["ReportSection"]))
        log_data = [["Operation", "Affected", "Details"]]
        log_data.extend([
            [
                Paragraph(_pdf_text(item.get("operation", "")), styles["SmallCell"]),
                str(item.get("affected", "")),
                Paragraph(_pdf_text(item.get("details", "")), styles["SmallCell"]),
            ]
            for item in cleaning_log
        ])
        log_table = Table(log_data, colWidths=[1.7 * inch, 0.8 * inch, 3.8 * inch], repeatRows=1)
        log_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(log_table)

    for title, table_data in (analysis_tables or {}).items():
        if table_data is None or table_data.empty:
            continue
        story.append(Paragraph(_pdf_text(title), styles["ReportSection"]))
        rows = [[Paragraph(_pdf_text(column), styles["SmallCell"]) for column in table_data.columns]]
        rows.extend([
            [Paragraph(_pdf_text(value), styles["SmallCell"]) for value in row]
            for row in table_data.astype(str).itertuples(index=False, name=None)
        ])
        width = page_width - doc.leftMargin - doc.rightMargin
        col_width = width / max(1, len(rows[0]))
        result_table = Table(rows, colWidths=[col_width] * len(rows[0]), repeatRows=1)
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(result_table)

    story.append(Paragraph("Statistical findings", styles["ReportSection"]))
    if findings:
        for index, finding in enumerate(findings, start=1):
            details = [
                f"<b>Evidence:</b> {_pdf_text(finding.evidence)}",
                f"<b>Uncertainty:</b> {_pdf_text(finding.uncertainty)}",
                f"<b>Limitations:</b> {_pdf_text('; '.join(finding.limitations))}",
                f"<b>Suggested next step:</b> {_pdf_text(finding.next_step)}",
            ]
            if finding.p_adj is not None:
                details.append(f"<b>Adjusted p-value:</b> {finding.p_adj:.4g}")
            story.append(KeepTogether([
                Paragraph(f"{index}. {_pdf_text(finding.claim)}", styles["FindingTitle"]),
                Paragraph("<br/>".join(details), styles["FindingBody"]),
                Spacer(1, 6),
            ]))
    else:
        story.append(Paragraph("No statistically supported findings were returned.", styles["BodyText"]))

    story.extend([
        PageBreak(),
        Paragraph("Methods and interpretation", styles["ReportSection"]),
        Paragraph(
            "Calculations are performed by deterministic Python analysis tools. Reported statistical "
            "associations do not establish causality. Results depend on the available data, selected "
            "analyses, assumptions, and missing-value handling. Review the evidence, uncertainty, and "
            "limitations before making decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph(
            "Privacy note: this report contains dataset-derived summaries. Review it for sensitive "
            "information before sharing.",
            styles["BodyText"],
        ),
    ])

    def draw_page(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(BLUE)
        canvas.rect(0, letter[1] - 0.23 * inch, page_width, 0.23 * inch, stroke=0, fill=1)
        _Footer(f"FlyBrain AutoAnalyst | {name}").drawOn(
            canvas, doc.leftMargin, 0.1 * inch
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()


def to_html(name, profile, quality, findings) -> str:
    q = "".join(f"<li><b>{html.escape(i['severity'])}</b>: {html.escape(i['detail'])}</li>" for i in quality) or "<li>No issues found.</li>"
    f = "".join(
        f"<div style='border:1px solid #ccc;padding:10px;margin:10px 0'><h3>{html.escape(x.claim)}</h3>"
        f"<p><b>Evidence:</b> {html.escape(str(x.evidence))}</p><p><b>Uncertainty:</b> {html.escape(x.uncertainty)}</p>"
        f"<p><b>Limitations:</b> {html.escape('; '.join(x.limitations))}</p><p><b>Next step:</b> {html.escape(x.next_step)}</p></div>"
        for x in findings) or "<p>No statistically solid findings.</p>"
    return (f"<html><body style='font-family:sans-serif;max-width:800px;margin:auto'><h1>Report: {html.escape(name)}</h1>"
            f"<p>{profile['n_rows']} rows, {profile['n_cols']} columns</p><h2>Data quality</h2><ul>{q}</ul><h2>Findings</h2>{f}</body></html>")
