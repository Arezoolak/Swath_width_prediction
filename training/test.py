
import torch
import torch.nn as nn
import numpy as np
from transformer import SwathWidthTransformer
import pandas as pd
import matplotlib.pyplot as plt

# Load saved test data
data = torch.load('/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/test_data.pt')
features = data['features']  # shape [N, 25, 512]
labels = data['labels']      # shape [N, 2]

# Load model
model = SwathWidthTransformer(feature_dim=512, num_frames=25)
model.load_state_dict(torch.load('/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/best_model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
features = features.to(device)
labels = labels.to(device)

with torch.no_grad():
    predictions = model(features)  # shape [N, 2]


# Move to CPU for numpy calculations
pred_np = predictions.cpu().numpy()  # shape [N, 2]
true_np = labels.cpu().numpy()       # shape [N, 2]

# MAE
mae = np.mean(np.abs(pred_np - true_np), axis=0)
print(f"🔹 MAE  : {mae}")

# RMSE
rmse = np.sqrt(np.mean((pred_np - true_np) ** 2, axis=0))
print(f"🔹 RMSE : {rmse}")

# Bias
bias = np.mean((pred_np - true_np)**2, axis=0)
print(f"🔹 Bias : {bias}")

# Variance of predictions
variance = np.var(pred_np, axis=0)
print(f"🔹 Variance of predictions: {variance}")



df = pd.DataFrame({
    'true_width': true_np,
    'pred_width': pred_np,

})

df.to_csv('/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/test_results.csv', index=False)
print("✅ Saved per-sample predictions to test_results.csv")


plt.scatter(true_np, pred_np, alpha=0.7)
plt.plot([min(true_np), max(true_np)], [min(true_np), max(true_np)], 'r--')
plt.xlabel('True Swath Width')
plt.ylabel('Predicted Swath Width')
plt.title('Prediction vs True Swath Width')
plt.grid(True)
plt.savefig("/home/arezou/UBONTO/my_own_projects/pytorch/pytorch-cnn/practice/classification_bird/last/scatter_plot_test.png")
plt.show() 













