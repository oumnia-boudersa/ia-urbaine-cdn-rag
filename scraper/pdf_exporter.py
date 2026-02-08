# pdf_exporter.py
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def save_to_pdf(results, filename="results.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for i, card in enumerate(results, 1):

        details = card["details"].replace("\n", "<br/>") if card["details"] else ""
        website = card.get('Site internet', 'N/A')
        text = (
            f"<b>Organisation {i}: {card['Organisation']}</b><br/>"
            f"<b>URL:</b> {card['url']}<br/>"
            f"<b>Site internet:</b> {card['Site internet']}<br/><br/>"
            f"Description: {card['details'] or 'N/A'}"
        )
        story.append(Paragraph(text, styles["Normal"]))
        story.append(PageBreak())

    doc.build(story)
