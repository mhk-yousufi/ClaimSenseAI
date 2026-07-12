import google.generativeai as genai
import os

# Securely grab the key from the terminal environment
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=api_key)

def analyze_claim(ocr_text, cnn_severity, fraud_score, tabular_data):
    # Using the latest 2026 model
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    prompt = f"""
    You are an AI Insurance Assistant. Review the following claim data and generate a professional summary.
    
    1. Tabular Data: {tabular_data}
    2. Computer Vision Analysis: {cnn_severity}
    3. Fraud Risk Score: {fraud_score}%
    4. Extracted Document Text: {ocr_text[:1000]}...
    
    Task:
    - Summarize the incident.
    - Note any discrepancies between the image severity and the claim amount.
    - Recommend whether the human officer should Approve, Reject, or Investigate further.
    Keep it under 150 words.
    """
    
    response = model.generate_content(prompt)
    return response.text