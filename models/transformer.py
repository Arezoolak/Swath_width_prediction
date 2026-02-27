import torch
import torch.nn as nn

class SwathWidthTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_frames=25, num_layers=2, num_heads=8, hidden_dim=512):
        super(SwathWidthTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, num_frames, feature_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True  # Important: batch is first dimension
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: 1 scalar (swath width)
        )

    def forward(self, x):
        """
        x shape: [batch_size, num_frames, feature_dim]
        """
        x = x + self.positional_encoding  # Add positional information
        x = self.transformer_encoder(x)   # Transformer processes the sequence
        
        # Pooling: take the mean across frames
        x = x.mean(dim=1)  # [batch_size, feature_dim]
        
        output = self.regressor(x)  # [batch_size, 1]
        
        return output.squeeze(1)  # Return as [batch_size]

