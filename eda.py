import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target  # Features and labels
feature_names = data.feature_names

# Select only the first 10 features (mean values)
X = X[:, :10]  

# Set up the figure for multiple histograms
fig, axes = plt.subplots(5, 2, figsize=(20, 18))
axes = axes.ravel()

# Plot histograms for each feature
for i in range(10):
    sns.histplot(X[y == 0, i], bins=30, color='blue', alpha=0.5, label="Benign", ax=axes[i])
    sns.histplot(X[y == 1, i], bins=30, color='red', alpha=0.5, label="Malignant", ax=axes[i])
    axes[i].set_title(feature_names[i])
    axes[i].legend()

plt.tight_layout()
plt.show()