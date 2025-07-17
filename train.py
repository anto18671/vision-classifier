from torchvision.datasets.folder import default_loader
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from torch import nn, optim
import albumentations as A
from tqdm import tqdm
import numpy as np
import torch
import timm
import os

# Custom dataset wrapper to apply Albumentations
class AlbumentationsDataset(Dataset):
    # Initialize the dataset
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    # Return length of dataset
    def __len__(self):
        return len(self.dataset)

    # Return transformed item
    def __getitem__(self, idx):
        # Load image and target
        path, target = self.dataset.samples[idx]
        image = default_loader(path)

        # Apply Albumentations transform
        image = self.transform(image=np.array(image))["image"]

        # Return transformed image and target
        return image, target

# Load training and validation datasets
def load_datasets(data_root, image_size, batch_size):
    # Define Albumentations training transform
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.0), contrast_limit=0.0, p=0.75),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Define Albumentations validation transform
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Load base datasets
    base_train = datasets.ImageFolder(root=os.path.join(data_root, "train"))
    base_val = datasets.ImageFolder(root=os.path.join(data_root, "val"))

    # Wrap datasets with Albumentations
    train_dataset = AlbumentationsDataset(base_train, train_transform)
    val_dataset = AlbumentationsDataset(base_val, val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)

    # Return datasets and loaders
    return base_train, train_loader, val_loader

# Create classification model using timm
def create_model(num_classes):
    # Load model with pretrained weights
    model = timm.create_model("regnety_006", pretrained=True, drop_rate=0.5)

    # Replace classifier head with correct number of classes
    model.reset_classifier(num_classes=num_classes)

    # Return model
    return model

# Train the model
def train_one_epoch(model, loader, optimizer, criterion, device):
    # Set model to training mode
    model.train()

    # Initialize loss accumulator
    total_loss = 0.0

    # Create tqdm loop
    progress = tqdm(loader, desc="Training")

    # Iterate over training data
    for i, (inputs, targets) in enumerate(progress):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update tqdm bar with average loss
        avg_loss = total_loss / (i + 1)
        progress.set_postfix(avg_loss=f"{avg_loss:.4f}")

    # Return final average loss
    return total_loss / len(loader)

# Validate the model
def validate(model, loader, criterion, device):
    # Set model to evaluation mode
    model.eval()

    # Initialize loss and accuracy counters
    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation
    with torch.no_grad():
        # Create tqdm loop
        progress = tqdm(loader, desc="Validation")

        # Iterate over validation data
        for i, (inputs, targets) in enumerate(progress):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Update tqdm bar with average loss
            avg_loss = total_loss / (i + 1)
            progress.set_postfix(avg_loss=f"{avg_loss:.4f}")

    # Compute accuracy
    accuracy = correct / total

    # Return validation loss and accuracy
    return total_loss / len(loader), accuracy

# Plot training and validation loss
def plot_losses(train_losses, val_losses, output_dir = "assets"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create figure and plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

# Define main function
def main():
    # Set hyperparameters
    data_root = "data"
    image_size = 28
    batch_size = 48
    num_epochs = 24
    learning_rate = 5e-4
    weight_decay = 1e-3

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset, train_loader, val_loader = load_datasets(data_root, image_size, batch_size)
    num_classes = len(train_dataset.classes)

    # Create model
    model = create_model(num_classes)
    model = model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize loss history
    train_losses = []
    val_losses = []

    # Train for multiple epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Record loss values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot and save loss curves
    plot_losses(train_losses, val_losses)

# Call the main function
if __name__ == "__main__":
    main()
