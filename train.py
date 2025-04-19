import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from copy import deepcopy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
train_data = datasets.ImageFolder('D:/Tumor Detection Model/dataset/train', transform=train_transform)
val_data = datasets.ImageFolder('D:/Tumor Detection Model/dataset/val', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load EfficientNet-B0
model = models.efficientnet_b1(pretrained=True)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify classifier with Dropout
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

# Training loop with early stopping
num_epochs = 10
best_acc = 0
patience = 3
trigger_times = 0
best_model = deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total * 100
    val_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    scheduler.step(val_acc)

    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⏹️ Early stopping triggered!")
            break

# Save best model
model.load_state_dict(best_model)
torch.save(model.state_dict(), "efficientnet_tumor_classifier.pth")
print(f"✅ Training complete. Best Validation Accuracy: {best_acc:.2f}%")
