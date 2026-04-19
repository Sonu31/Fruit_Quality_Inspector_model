# train.py
# Yeh file model ko train karegi

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
# ✅ Sahi — ImageFolder torchvision mein hota hai!
from torchvision.datasets import ImageFolder

print("="*50)
print("   DEFECT DETECTION - MODEL TRAINING")
print("="*50)

# ================================
# 1. DEVICE SETUP
# GPU hai toh GPU use karo, nahi toh CPU
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ================================
# 2. IMAGE TRANSFORM
# Images ko model ke liye ready karo
# ================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Size fix karo
    transforms.RandomHorizontalFlip(),      # Mirror karo (variety ke liye)
    transforms.RandomRotation(15),          # Thoda ghumao
    transforms.ColorJitter(brightness=0.3), # Brightness change karo
    transforms.ToTensor(),                  # Numbers mein badlo
    transforms.Normalize(                   # Normalize karo
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# 3. DATASET LOAD KARO
# ================================
print("\nDataset load ho raha hai...")

train_dataset = ImageFolder('dataset/train', transform=train_transform)
test_dataset  = ImageFolder('dataset/test',  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"Train images : {len(train_dataset)}")
print(f"Test images  : {len(test_dataset)}")
print(f"Classes      : {train_dataset.classes}")
# Output: ['defective', 'normal']

# ================================
# 4. MODEL BANAO
# ResNet50 use karenge — pehle se trained hai
# ================================
print("\nModel ban raha hai...")

model = models.resnet50(pretrained=True)

# Last layer change karo
# ResNet ka last layer 1000 classes ke liye tha
# Hum sirf 2 chahte hain: Normal / Defective
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.4),        # Overfitting rokne ke liye
    nn.Linear(256, 2)       # 2 = [Normal, Defective]
)

model = model.to(device)
print("Model ready!")

# ================================
# 5. TRAINING SETUP
# ================================
criterion = nn.CrossEntropyLoss()   # Loss function
optimizer = torch.optim.Adam(       # Optimizer
    model.parameters(), 
    lr=0.001
)

# Har 5 epoch baad learning rate kam karo
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.1
)

# ================================
# 6. TRAINING LOOP
# ================================
print("\n" + "="*50)
print("TRAINING SHURU...")
print("="*50)

EPOCHS = 10
train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        # Progress dikhao
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")
    
    # Epoch stats
    epoch_loss = total_loss / len(train_loader)
    epoch_acc  = correct / total * 100
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    scheduler.step()
    
    print(f"\nEpoch {epoch+1} Complete!")
    print(f"Loss    : {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_acc:.2f}%")
    print("-"*40)

# ================================
# 7. MODEL TEST KARO
# ================================
print("\nModel test ho raha hai...")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Report print karo
print("\n" + "="*50)
print("TEST RESULTS:")
print("="*50)
print(classification_report(
    all_labels, 
    all_preds,
    target_names=['Defective', 'Normal']
))

# ================================
# 8. MODEL SAVE KARO
# ================================
torch.save(model.state_dict(), 'defect_model.pth')
print("\nModel saved: defect_model.pth")
print("Ab STEP 2 ke liye ready ho!")