# 🛡️ ClaimSense AI – Intelligent Insurance Co-Pilot

An end-to-end multimodal AI application designed to accelerate insurance claim processing using Deep Learning, Explainable AI (XAI), and Large Language Models.

## 🚀 The Business Problem
Traditional insurance claims take days to process due to manual data entry, PDF reading, and image inspection. ClaimSense AI acts as a "Co-Pilot," reducing processing time from days to minutes while keeping a Human-in-the-Loop for final compliance.

## 🧠 Multimodal AI Pipeline
This project exceeds the hackathon requirements by utilizing 4 data modalities:
1. **Images (Computer Vision):** ResNet/YOLOv8 detects damage severity.
2. **Tabular Data (Machine Learning):** Random Forest predicts fraud risk based on user history.
3. **PDF Documents (OCR):** PyMuPDF extracts text from policies and police reports.
4. **Text Reasoning (LLM):** Gemini Pro correlates image severity, fraud risk, and policy text to generate an actionable recommendation.

## 🔍 Explainable AI (XAI)
To ensure trust and transparency for the claims officer:
* **Grad-CAM:** Generates heatmaps highlighting the exact pixels the CNN used to determine vehicle damage.
* **SHAP:** Visualizes feature importance, explaining *why* the ML model assigned a specific fraud risk score.

## ⚙️ Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/mhk-yousufi/ClaimSenseAI.git](https://github.com/mhk-yousufi/ClaimSenseAI.git)
cd ClaimSenseAI

# Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the application
$env:GEMINI_API_KEY="your_api_key_here"
streamlit run app.py
