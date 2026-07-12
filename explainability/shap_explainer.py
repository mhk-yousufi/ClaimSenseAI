import shap
import matplotlib.pyplot as plt
import numpy as np

def generate_shap_plot(model, features, output_path="uploads/shap_plot.png"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # For RandomForest classifiers, SHAP returns a list of arrays (one for each class).
    # We want to explain the "Fraud" class, which is index 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(8, 4))
    
    # FIX: Convert feature names to a NumPy array to prevent the indexing TypeError
    feature_names = np.array(["Claim Amount", "Driver Age", "Prev Claims", "Vehicle Age"])
    
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path