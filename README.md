# Swath_width_prediction
Hybrid CNN-Transformer model for real-time swath width estimation


## Pretrained Model Weights

Pretrained models for both field-of-view configurations are available at:

Zenodo DOI: https://doi.org/XXXX

Included weights:
- narrow_fov_model.pth
- wide_fov_model.pth

Download the weights and place them inside:

models/narrow_fov/
models/wide_fov/

Example usage:

python inference/predict_real_images.py --weights models/narrow_fov/narrow_fov_model.pth
