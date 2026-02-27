# Narrow Field-of-View (FOV) Model

This folder corresponds to the Transformer-based deep learning model 
trained using the **narrow FOV synthetic dataset**, where only a 
partial fertilizer spread pattern is visible in each frame.

## Model Description

- CNN Backbone: ResNet-18
- Temporal Modeling: Transformer Encoder
- Input: 25 sequential frames
- Output: Predicted swath width (meters)

## Training Configuration

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 30
- Dataset: Synthetic narrow FOV videos

## Pretrained Weights

The pretrained model weights are available at:

[Zenodo DOI lin]

## Usage

After downloading the pretrained weights (best_model.pth), you can use it for prediction, if you prefer narrow Fov and closer camera to the spreader.


