#%% 
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
plt.savefig("out/out.pdf")

# %%
import pandas as pd
X = data.data; y = data.target; feature_names = data.feature_names
XX = pd.DataFrame(X, columns = feature_names)
XX.describe()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

X = data.data; y = data.target; feature_names = data.feature_names
XX = pd.DataFrame(X, columns = feature_names)[['mean area', 'mean perimeter']] 

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size = 0.3, random_state=42)
sc = StandardScaler().fit(X_train)
X_train_norm = sc.transform(X_train)
X_test_norm = sc.transform(X_test)
lr = LogisticRegression().fit(X_train_norm, y_train)
lr.coef_

# %%
type(X), type(y)
np.savez("./breast-cancer.npz", X=X, y=y, feature_names = feature_names)
# %%
import pandas as pd
X = data.data
df = pd.DataFrame(X, columns = ['_'.join(x.split()) for x in feature_names])
df['label'] = y
df.to_csv("breast-cancer-data.csv")
# %%
df.to