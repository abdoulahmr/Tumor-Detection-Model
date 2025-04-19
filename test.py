import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- SETTINGS ---
model_path = 'efficientnet_tumor_classifier.pth'
test_dir = 'D:/Tumor Detection Model/dataset/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

# --- TRANSFORMS (must match training transforms) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD TEST DATA ---
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- LOAD MODEL ---
from torchvision.models import efficientnet_b0
model = efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- EVALUATE ---
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n✅ Accuracy on test set: {accuracy:.2f}%")
