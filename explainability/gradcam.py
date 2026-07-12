import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_gradcam_heatmap(image_path, output_path="uploads/heatmap.jpg"):
    """
    Mock Grad-CAM generator for the prototype.
    In production, this attaches PyTorch hooks to the last CNN layer.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    
    # Generate a synthetic heatmap focused on the center/lower half (typical damage area)
    heatmap = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(heatmap, (112, 150), 60, (1.0,), -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)
    return output_path
