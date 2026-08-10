import os
import cv2  # OpenCV for preprocessing
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Step 3.1: Define Preprocessing Transforms
# This resizes, enhances contrast, normalizes, and augments images
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL for transforms
    transforms.Resize((224, 224)),  # Resize to 224x224 for DenseNet
    transforms.RandomHorizontalFlip(),  # Augment: flip horizontally
    transforms.RandomRotation(15),  # Augment: rotate up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Augment: adjust brightness/contrast
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
])

# Custom function for additional preprocessing (denoising and CLAHE)
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image with OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # Denoising: Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # CLAHE for contrast enhancement (on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    # Merge back to color (simple way: replace value channel in HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = enhanced_gray
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

# Step 3.2: Custom Dataset Class
class FundusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file,usecols=[2, 3],           # ← Column 2 = image name, Column 3 = level
            names=['image', 'level'], # ← Give them proper names (skip header row)
            header=0,                 # ← Your CSV HAS a header row ("Unnamed image", "level")
            dtype={'image': str})  # Load your CSV (change to pd.read_excel if .xlsx)
        self.root_dir = root_dir
        self.transform = transform
        self.extension = ".jpeg"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base_name = self.labels.iloc[idx, 0]                     # e.g. "0" or "32968"
        filename = base_name + self.extension                    # ← add .jpg here
        img_name = os.path.join(self.root_dir, filename)
        image = preprocess_image(img_name)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_name}")
        label = int(self.labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 3.3: Load and Split Dataset
dataset = FundusDataset(
    csv_file='data/raw/labels.csv',          # path to your CSV from the main folder
    root_dir='data/raw/Images',              # path to the folder that contains all images
    transform=transform
)
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)  # 80/20 split
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# DataLoaders (batches images for training)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)  # Adjust num_workers based on your CPU
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# Quick debug: check if we can load at least one image
print("\n=== Dataset Check ===")
print("Total images found in CSV:", len(dataset))

if len(dataset) > 0:
    sample_idx = 0
    filename = dataset.labels.iloc[sample_idx, 0] + ".jpeg"
    full_path = os.path.join(dataset.root_dir, filename)
    print("Sample image path:", full_path)
    
    if os.path.exists(full_path):
        print("→ Path exists ✓")
        img = cv2.imread(full_path)
        if img is not None:
            print("→ Image loaded successfully (shape:", img.shape, ")")
        else:
            print("→ cv2.imread failed – file may be corrupted or wrong format")
    else:
        print("→ Path does NOT exist ❌ – check CSV filenames vs actual files")
else:
    print("→ No images found in dataset – check CSV file")

# Optional: limit to small subset for fast testing
# from torch.utils.data import Subset
# dataset = Subset(dataset, list(range(min(500, len(dataset)))))

import torch.nn as nn
import torchvision.models as models

# Step 4.1: Load Pre-trained DenseNet-201
model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1) # Pre-trained on ImageNet
num_features = model.classifier.in_features  # Get input size of last layer
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Prevent overfitting
    nn.Linear(num_features, 5)  # 5 classes (0-4 DR levels)
)

# Move to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4.2: Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Learning rate
criterion = nn.CrossEntropyLoss()  # For multi-class

# Step 4.3: Training Loop
num_epochs = 1  # Start with 20; monitor for overfitting
for epoch in range(num_epochs):
    model.train()  # Training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backprop
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation (simple accuracy check on test set)
    model.eval()  # Evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Step 4.4: Save the Trained Model
torch.save(model.state_dict(), 'dr_model.pth')
print("Model saved as dr_model.pth")

# Step 5: Inference Function (for user-uploaded image)
def predict_dr(image_path):
    model.load_state_dict(torch.load('dr_model.pth'))  # Load saved model
    model.eval()  # Eval mode
    image = preprocess_image(image_path)  # Custom preprocess
    image = transform(image).unsqueeze(0).to(device)  # Transform and add batch dim
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    levels = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"}
    return levels[predicted.item()]

# Example usage (uncomment to test)
# result = predict_dr('path/to/new_fundus_image.jpg')
# print(f"Predicted: {result}")

