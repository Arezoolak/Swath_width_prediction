import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.transformer_encoder import SwathWidthTransformer


# =======================================================
# ARGUMENT PARSER
# =======================================================
parser = argparse.ArgumentParser(description="Evaluate Swath Width Transformer Model")

parser.add_argument('--test_data', type=str, default='outputs/test_data.pt',
                    help='Path to saved test_data.pt')
parser.add_argument('--model_path', type=str, default='outputs/best_model.pth',
                    help='Path to trained model weights')
parser.add_argument('--output_dir', type=str, default='evaluation_results',
                    help='Directory to save results')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# =======================================================
# LOAD TEST DATA
# =======================================================
data = torch.load(args.test_data)

features = data['features']  # shape [N, 25, 512]
labels = data['labels']

# =======================================================
# LOAD MODEL
# =======================================================
model = SwathWidthTransformer(feature_dim=512, num_frames=25)

model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

features = features.to(device)
labels = labels.to(device)

# =======================================================
# PREDICTION
# =======================================================
with torch.no_grad():
    predictions = model(features)

pred_np = predictions.cpu().numpy()
true_np = labels.cpu().numpy()

# =======================================================
# METRICS
# =======================================================

# MAE
mae = np.mean(np.abs(pred_np - true_np))
print(f"🔹 MAE  : {mae:.4f}")

# RMSE
rmse = np.sqrt(np.mean((pred_np - true_np) ** 2))
print(f"🔹 RMSE : {rmse:.4f}")

# Bias (mean error, not squared)
bias = np.mean(pred_np - true_np)
print(f"🔹 Bias : {bias:.4f}")

# Variance of predictions
variance = np.var(pred_np)
print(f"🔹 Variance of predictions: {variance:.4f}")

# =======================================================
# SAVE PER-SAMPLE RESULTS
# =======================================================
df = pd.DataFrame({
    'true_width': true_np.flatten(),
    'pred_width': pred_np.flatten(),
    'error': (pred_np - true_np).flatten()
})

csv_path = os.path.join(args.output_dir, "test_results.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved predictions to {csv_path}")

# =======================================================
# SCATTER PLOT
# =======================================================
plt.figure(figsize=(6, 6))
plt.scatter(true_np, pred_np, alpha=0.6)

min_val = min(true_np.min(), pred_np.min())
max_val = max(true_np.max(), pred_np.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel('True Swath Width (m)')
plt.ylabel('Predicted Swath Width (m)')
plt.title('Prediction vs True Swath Width')
plt.grid(True)

plot_path = os.path.join(args.output_dir, "scatter_plot_test.png")
plt.savefig(plot_path)
plt.close()

print(f"✅ Scatter plot saved to {plot_path}")

print("🎯 Evaluation completed successfully")












