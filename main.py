# api.py
# Yeh FastAPI server hai

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import io

# ================================
# 1. FastAPI App Banao
# ================================
app = FastAPI(title="Fruit Quality Inspector API")

# CORS — Android se call allow karo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 2. Model Load Karo
# ================================
class DefectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# Model load karo — server start hone pe ek baar
print("Model load ho raha hai...")
model = DefectModel()
# model.load_state_dict(torch.load('defect_model.pth', map_location='cpu'))

state_dict = torch.load('defect_model.pth', map_location='cpu')

# Keys automatically fix karo
new_state_dict = {}
for key, value in state_dict.items():
    if not key.startswith("model."):
        new_key = "model." + key
    else:
        new_key = key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)


model.eval()
print("✅ Model ready!")

# ================================
# 3. Image Transform
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# 4. API Routes
# ================================

# Home Route — Check karo API chal rahi hai
@app.get("/")
def home():
    return {
        "message": "✅ Fruit Quality Inspector API Running!",
        "version": "1.0"
    }

# Main Route — Image bhejo result lo
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Image read karo
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transform karo
        img_tensor = transform(image).unsqueeze(0)
        
        # Model se predict karo
        with torch.no_grad():
            output = model(img_tensor)
            prob   = torch.softmax(output, dim=1)
            pred   = output.argmax(1).item()
        
        # Result tayar karo
        classes    = ['Defective', 'Normal']
        result     = classes[pred]
        confidence = prob[0][pred].item() * 100
        
        # Response bhejo
        return {
            "status"     : "success",
            "result"     : result,
            "confidence" : round(confidence, 2),
            "message"    : "✅ Theek Hai!" if result == "Normal" 
                           else "❌ Kharab Hai!",
            "emoji"      : "✅" if result == "Normal" else "❌"
        }
    
    except Exception as e:
        return {
            "status"  : "error",
            "message" : str(e)
        }