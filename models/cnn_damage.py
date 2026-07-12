import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

class DamageModel:
    def __init__(self):
        # Loading a pre-trained ResNet18 for hackathon prototype speed
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.classes = ['Minor Damage', 'Moderate Damage', 'Severe Damage']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Mocking classification mapping to our 3 classes based on ResNet outputs
        confidence, predicted = torch.max(probabilities[:3], 0)
        return self.classes[predicted.item()], confidence.item() * 100