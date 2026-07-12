from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def generate_claim_report(claim_id, data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 750, f"ClaimSense AI - Official Report (Claim #{claim_id})")
    
    c.setFont("Helvetica", 12)
    y_pos = 700
    
    lines = [
        f"Vehicle: {data['vehicle_model']}",
        f"Original Claim Amount: ${data['claim_amount']}",
        f"Final Approved Amount: ${data['modified_amount']}",
        f"Status: {data['status']}",
        f"AI Damage Assessment: {data['damage_severity']} ({data['cnn_confidence']:.2f}% Confidence)",
        f"Fraud Risk Score: {data['fraud_score']:.2f}%",
        "",
        "LLM Reasoning Summary:"
    ]
    
    for line in lines:
        c.drawString(50, y_pos, line)
        y_pos -= 20
        
    # Write LLM text wrapping
    if data['llm_summary']:
        for text_line in data['llm_summary'].split('\n'):
            c.drawString(50, y_pos, text_line[:100]) # Basic text wrap for prototype
            y_pos -= 15
        
    c.save()
    return output_path