# predict.py
# Ek image do — result aayega

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys

# ================================
# 1. Model Load Karo
# ================================
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load('defect_model.pth', 
                          map_location='cpu'))
    model.eval()
    return model

# ================================
# 2. Image Predict Karo
# ================================
def predict(image_path):
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Image load karo
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    
    # Predict karo
    model = load_model()
    
    with torch.no_grad():
        output = model(img)
        prob   = torch.softmax(output, dim=1)
        pred   = output.argmax(1).item()
    
    # Result
    classes = ['Defective', 'Normal']
    result  = classes[pred]
    confidence = prob[0][pred].item() * 100
    
    # Print karo
    print("\n" + "="*40)
    print("   DEFECT DETECTION RESULT")
    print("="*40)
    print(f"Image     : {image_path}")
    
    if result == 'Defective':
        print(f"Result    : ❌ DEFECTIVE - Kharab Hai!")
    else:
        print(f"Result    : ✅ NORMAL - Theek Hai!")
    
    print(f"Confidence: {confidence:.2f}%")
    print("="*40)

# ================================
# 3. Run Karo
# ================================
# Command: python predict.py apple.jpg
if __name__ == "__main__":
    image_path = sys.argv[1]  # Image path do
    predict(image_path)