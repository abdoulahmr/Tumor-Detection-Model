import torch
from torchvision import models, transforms
from PIL import Image
import os

# --- SETTINGS ---
model_path = 'D:/Tumor Detection Model/model/efficientnet_tumor_classifier_97.92.pth'
image_path = 'D:/Tumor Detection Model/11 - Copy (2).png'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
from torchvision.models import efficientnet_b0
model = efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- LOAD & PREPROCESS IMAGE ---
img = Image.open(image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# --- PREDICT ---
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    predicted = torch.argmax(probs).item()

# --- INTERPRETATION ---
class_names = ['Normal', 'Tumor']
print(f'🧠 Prediction: {class_names[predicted]}')
for i, prob in enumerate(probs):
    print(f'📊 {class_names[i]}: {prob.item() * 100:.2f}%')
