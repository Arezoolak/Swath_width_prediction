from torchvision import transforms
from feature_extractor import CNNFeatureExtractor, FertilizerSpreadDataset
from transformer import SwathWidthTransformer  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =======================================================
# LOAD LABEL CSV
# =======================================================
labels_df = pd.read_csv(
    '/home/arezou/UBONTO/Dataset_swath_width/labels.csv'
)

# Train/Val/Test split
train_df, temp_df = train_test_split(labels_df, test_size=0.3, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# =======================================================
# HYPERPARAMETERS
# =======================================================
batch_size = 4
num_epochs = 30
learning_rate = 1e-4
num_frames = 25


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =======================================================
# INITIALIZE CNN FEATURE EXTRACTOR
# =======================================================
cnn_extractor = CNNFeatureExtractor(backbone='resnet18', pretrained=True)
cnn_extractor.eval()    # Freeze CNN during inference

# =======================================================
# DATASETS
# =======================================================
root_path = '/home/arezou/UBONTO/Dataset_swath_width'

train_dataset = FertilizerSpreadDataset(root_dir=root_path, labels_df=train_df,
                                        cnn_extractor=cnn_extractor, transform=transform)

val_dataset = FertilizerSpreadDataset(root_dir=root_path, labels_df=val_df,
                                      cnn_extractor=cnn_extractor, transform=transform)

test_dataset = FertilizerSpreadDataset(root_dir=root_path, labels_df=test_df,
                                       cnn_extractor=cnn_extractor, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# =======================================================
# SAVE TEST DATA (FEATURES + LABELS)
# =======================================================
all_features = []
all_labels = []

for features, labels in test_loader:
    # features shape: [B, 36, 512]
    # labels shape:   [B, 41]
    all_features.append(features)
    all_labels.append(labels)

features_tensor = torch.cat(all_features, dim=0)
labels_tensor   = torch.cat(all_labels,   dim=0)

torch.save({'features': features_tensor, 'labels': labels_tensor},
           '/home/arezou/UBONTO/test_data.pt')

print("✅ Test data saved to test_data.pt")


# =======================================================
# INITIALIZE TRANSFORMER MODEL
# =======================================================
model = SwathWidthTransformer(
    feature_dim=512,
    num_frames=num_frames,     # <-- 36 frames
            # <-- 41 output bins
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =======================================================
# TRAINING LOOP
# =======================================================
best_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        features = features.to(device)     # [B, 36, 512]
        labels   = labels.to(device)       # [B, 41]

        optimizer.zero_grad()
        outputs = model(features)          # [B, 41]

        loss = criterion(outputs, labels)  # vector regression loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {epoch_loss:.4f}")

    # ------------------ Validation ------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels   = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(),
                   '/home/arezou/UBONTO/best_model.pth')
        print(f"🔥 Best model saved (Val loss={val_loss:.4f})")

# =======================================================
# PLOT LOSS CURVES
# =======================================================
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("/home/arezou/UBONTO/loss_plot.png")
plt.show()
