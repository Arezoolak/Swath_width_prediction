import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from models.feature_extractor import CNNFeatureExtractor, FertilizerSpreadDataset
from models.transformer_encoder import SwathWidthTransformer


# =======================================================
# ARGUMENT PARSER
# =======================================================
parser = argparse.ArgumentParser(description="Train Swath Width Transformer Model")

parser.add_argument('--data_root', type=str, default='dataset',
                    help='Root directory of dataset (contains labels.csv and data folders)')
parser.add_argument('--output_dir', type=str, default='outputs',
                    help='Directory to save models and plots')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_frames', type=int, default=25)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# =======================================================
# LOAD LABEL CSV
# =======================================================
labels_path = os.path.join(args.data_root, "labels.csv")
labels_df = pd.read_csv(labels_path)

train_df, temp_df = train_test_split(labels_df, test_size=0.3,
                                     random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                   random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# =======================================================
# HYPERPARAMETERS
# =======================================================
batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.learning_rate
num_frames = args.num_frames

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =======================================================
# INITIALIZE CNN FEATURE EXTRACTOR
# =======================================================
cnn_extractor = CNNFeatureExtractor(backbone='resnet18', pretrained=True)
cnn_extractor.eval()

# =======================================================
# DATASETS
# =======================================================
train_dataset = FertilizerSpreadDataset(
    root_dir=args.data_root,
    labels_df=train_df,
    cnn_extractor=cnn_extractor,
    transform=transform
)

val_dataset = FertilizerSpreadDataset(
    root_dir=args.data_root,
    labels_df=val_df,
    cnn_extractor=cnn_extractor,
    transform=transform
)

test_dataset = FertilizerSpreadDataset(
    root_dir=args.data_root,
    labels_df=test_df,
    cnn_extractor=cnn_extractor,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =======================================================
# SAVE TEST DATA
# =======================================================
features_list = []
labels_list = []

for features, labels in test_loader:
    features_list.append(features)
    labels_list.append(labels)

features_tensor = torch.cat(features_list, dim=0)
labels_tensor   = torch.cat(labels_list, dim=0)

torch.save(
    {'features': features_tensor, 'labels': labels_tensor},
    os.path.join(args.output_dir, "test_data.pt")
)

print("✅ Test data saved")

# =======================================================
# INITIALIZE TRANSFORMER MODEL
# =======================================================
model = SwathWidthTransformer(
    feature_dim=512,
    num_frames=num_frames
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

    for features, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{num_epochs}"):

        features = features.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # ---------------- Validation ----------------
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
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} | Train: {epoch_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, "best_model.pth"))
        print("🔥 Best model saved")

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
plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
plt.close()

print("✅ Training completed")
