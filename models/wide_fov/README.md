
---

# ✅ models/wide_fov/README.md

```markdown
# Wide Field-of-View (FOV) Model

This folder corresponds to the Transformer-based deep learning model 
trained using the **wide FOV synthetic dataset**, where the full 
fertilizer spread pattern is visible.

## Model Description

- CNN Backbone: ResNet-18
- Temporal Modeling: Transformer Encoder
- Input: 25 sequential frames
- Output: Predicted swath width (meters)

## Training Configuration

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 30
- Dataset: Synthetic wide FOV videos

## Pretrained Weights

The pretrained model weights are available at:

[Zenodo DOI link]

## Usage

After downloading the pretrained weights (best.model.pth), you can use it for prediction, if you prefer wide Fov and farther camera from spreader (full coverage of distribution).
