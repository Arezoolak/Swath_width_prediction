import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        if backbone == 'resnet18':
            cnn = models.resnet18(pretrained=pretrained)
            self.feature_dim = cnn.fc.in_features  # 512 for ResNet18
            cnn = nn.Sequential(*list(cnn.children())[:-1])  # Remove the final fc layer
            self.backbone = cnn

        elif backbone == 'efficientnet_b0':
            cnn = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = cnn.classifier[1].in_features  # 1280 for EfficientNet-B0
            cnn.classifier = nn.Identity()  # Remove the final classifier
            self.backbone = cnn

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        # x shape: [batch_size, 3, 512, 512]
        features = self.backbone(x)  # Feature shape [batch_size, feature_dim, 1, 1]
        if features.dim() == 4:
            features = features.squeeze(-1).squeeze(-1)  # Remove extra dimensions
        return features  # Shape: [batch_size, feature_dim]



class FertilizerSpreadDataset(Dataset):
    def __init__(self, root_dir, labels_df, cnn_extractor, transform=None):
        """
        labels_df: dataframe directly instead of reading CSV here
        """
        self.root_dir = root_dir
        self.labels_df = labels_df.reset_index(drop=True)  # Reset index
        self.cnn_extractor = cnn_extractor
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        folder_name = str(int(self.labels_df.iloc[idx]['folder']))  # Important: int cast
        label = self.labels_df.iloc[idx]['label']
        
        folder_path = os.path.join(self.root_dir, folder_name)
        frame_files = sorted(os.listdir(folder_path))
        
        features = []
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            with torch.no_grad():
                feature = self.cnn_extractor(image.unsqueeze(0))
                feature = feature.squeeze(0)
            features.append(feature)
        
        features = torch.stack(features, dim=0)  # [25, 512]
        
        return features, torch.tensor(label, dtype=torch.float32)

