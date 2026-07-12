# app.py
import streamlit as st
import os
import time
from database.db import init_db, insert_claim, update_claim, get_all_claims
from models.cnn_damage import DamageModel
from models.fraud_model import FraudDetector
from ocr.pdf_reader import extract_text_from_pdf
from llm.report_generator import analyze_claim
from explainability.gradcam import generate_gradcam_heatmap
from explainability.shap_explainer import generate_shap_plot
from reports.pdf_gen import generate_claim_report

# Initialize
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
init_db()

st.set_page_config(page_title="ClaimSense AI", layout="wide", page_icon="🛡️")

st.title("🛡️ ClaimSense AI – Intelligent Insurance Co-Pilot")

# Load Models
@st.cache_resource
def load_models():
    return DamageModel(), FraudDetector()

cnn_model, fraud_model = load_models()

tab1, tab2, tab3 = st.tabs(["📄 Submit New Claim", "👨‍⚖️ Human-in-the-Loop Review", "📊 Dashboard & Reports"])

# --- TAB 1: SUBMIT CLAIM ---
with tab1:
    st.header("Step 1: Multimodal Data Ingestion")
    
    col1, col2 = st.columns(2)
    with col1:
        vehicle = st.text_input("Vehicle Model", "Toyota Camry 2020")
        claim_amount = st.number_input("Claim Amount ($)", 1000)
        driver_age = st.number_input("Driver Age", 35)
        prev_claims = st.number_input("Previous Claims", 0)
        vehicle_age = st.number_input("Vehicle Age (Years)", 4)
        
    with col2:
        image_file = st.file_uploader("Upload Car Damage Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
        pdf_file = st.file_uploader("Upload Police Report / Policy (PDF)", type=['pdf'])
        
    if st.button("Process Claim via AI Pipeline"):
        if image_file and pdf_file:
            with st.spinner("Executing Multimodal AI Pipeline..."):
                # Save files
                img_path = os.path.join("uploads", image_file.name)
                pdf_path = os.path.join("uploads", pdf_file.name)
                with open(img_path, "wb") as f: f.write(image_file.getbuffer())
                with open(pdf_path, "wb") as f: f.write(pdf_file.getbuffer())
                
                # 1. CNN Image Analysis
                severity, cnn_conf = cnn_model.predict(img_path)
                generate_gradcam_heatmap(img_path, "uploads/current_heatmap.jpg")
                
                # 2. OCR Processing
                extracted_text = extract_text_from_pdf(pdf_path)
                
                # 3. Fraud Detection & SHAP
                fraud_score, features = fraud_model.predict_risk(claim_amount, driver_age, prev_claims, vehicle_age)
                generate_shap_plot(fraud_model.model, features, "uploads/current_shap.png")
                
                # 4. LLM Reasoning
                tabular_data = f"{vehicle}, Amount: ${claim_amount}, Age: {driver_age}"
                llm_summary = analyze_claim(extracted_text, severity, fraud_score, tabular_data)
                
                # Database Insert
                data = {
                    "vehicle_model": vehicle, "claim_amount": claim_amount,
                    "driver_age": driver_age, "previous_claims": prev_claims,
                    "damage_severity": severity, "cnn_confidence": cnn_conf,
                    "fraud_score": fraud_score, "llm_summary": llm_summary
                }
                insert_claim(data)
                
            st.success("Pipeline Complete! Claim routed to Human Review panel.")
        else:
            st.error("Please upload both an Image and a PDF document.")


# --- TAB 2: HUMAN IN THE LOOP ---
with tab2:
    st.header("Step 2: Explainable AI & Human Decision")
    
    claims_df = get_all_claims()
    pending_claims = claims_df[claims_df['status'] == 'Pending Review']
    
    if not pending_claims.empty:
        claim_to_review = st.selectbox("Select Claim to Review", pending_claims['claim_id'])
        claim_data = pending_claims[pending_claims['claim_id'] == claim_to_review].iloc[0]
        
        st.markdown(f"### Claim #{claim_to_review} - {claim_data['vehicle_model']}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Requested Amount", f"${claim_data['claim_amount']}")
        col2.metric("AI Damage Class", claim_data['damage_severity'])
        col3.metric("Fraud Risk Score", f"{claim_data['fraud_score']:.1f}%")
        
        st.markdown("---")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.subheader("Visual Evidence (Grad-CAM)")
            st.image("uploads/current_heatmap.jpg", caption=f"CNN Confidence: {claim_data['cnn_confidence']:.1f}%")
            
        with exp_col2:
            st.subheader("Tabular Risk Factors (SHAP)")
            st.image("uploads/current_shap.png", caption="Feature contributions to Fraud Score")
            
        st.markdown("### LLM Assistant Recommendation")
        st.info(claim_data['llm_summary'])
        
        st.markdown("### Human Officer Decision")
        modified_amount = st.number_input("Modify Amount (if necessary)", value=float(claim_data['claim_amount']))
        
        dec_col1, dec_col2, dec_col3 = st.columns(3)
        if dec_col1.button("✅ Approve Claim", use_container_width=True):
            update_claim(claim_to_review, "Approved", modified_amount)
            st.success("Claim Approved!")
            st.rerun()
            
        if dec_col2.button("❌ Reject Claim", use_container_width=True):
            update_claim(claim_to_review, "Rejected", 0)
            st.error("Claim Rejected!")
            st.rerun()
            
        if dec_col3.button("⚠️ Request Investigation", use_container_width=True):
            update_claim(claim_to_review, "Under Investigation", modified_amount)
            st.warning("Claim Sent for Investigation!")
            st.rerun()
            
    else:
        st.write("No pending claims to review.")

# --- TAB 3: DASHBOARD & REPORTS ---
with tab3:
    st.header("Step 3: Database & Export")
    
    claims_df = get_all_claims()
    st.dataframe(claims_df, use_container_width=True)
    
    st.subheader("Download Official Report")
    reviewed_claims = claims_df[claims_df['status'] != 'Pending Review']
    
    if not reviewed_claims.empty:
        export_claim = st.selectbox("Select Reviewed Claim", reviewed_claims['claim_id'])
        
        if st.button("Generate PDF Report"):
            report_data = claims_df[claims_df['claim_id'] == export_claim].iloc[0]
            report_path = f"reports/claim_{export_claim}_report.pdf"
            generate_claim_report(export_claim, report_data, report_path)
            
            with open(report_path, "rb") as pdf_file:
                st.download_button("📥 Download PDF", data=pdf_file, file_name=f"Claim_{export_claim}_Report.pdf", mime="application/pdf")
    else:
        st.write("Process and review a claim to generate reports.")