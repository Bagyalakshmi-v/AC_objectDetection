
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split

# Defining parameters
input_dir = '/data'            #Ensure the path as absolute in local execution
brands = ['godrej', 'panasonic', 'lg', 'voltas', 'bluestar']
num_classes = len(brands)  # Number of brands

# Defining transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Dataset Class
class ACDataset(Dataset):
    def __init__(self, root_dir, brands, transform=None):
        self.root_dir = root_dir
        self.brands = brands
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, brand in enumerate(brands):
            brand_dir = os.path.join(root_dir, brand)
            if os.path.isdir(brand_dir):
                for img_file in os.listdir(brand_dir):
                    self.image_paths.append(os.path.join(brand_dir, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Creating datasets and dataloaders
full_dataset = ACDataset(input_dir, brands, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet and adjust the final layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for the number of classes

# Accessing GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'ac_brand_model.pth')


# Testing Function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total}%')


test_model(model, test_loader)


# Function to predict on a single image
def predict_image(image_path, model, transform, brands):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return brands[predicted.item()]


# Example usage of single-image prediction
image_path = '000001.jpg'  # Replace with a test image path
predicted_brand = predict_image(image_path, model, transform, brands)
print(f'Predicted brand: {predicted_brand}')
