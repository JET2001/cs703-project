# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target  # Features and labels
X = X[:, [1,2, 4,8, 9]]
print(X.shape)
# print(X.columns)
# %%
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities for AUC calculation
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1

fpr, tpr, _ = roc_curve(y_test, y_prob)
# Compute AUC
auc = roc_auc_score(y_test, y_prob)

# %%
print(f"Logistic Regression AUC: {auc:.4f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression (Breast Cancer Dataset)")
plt.legend()
plt.grid()
plt.show()
# %%
print("coef = ", model.coef_)
print("intercept = ", model.intercept_)
# %%
# model.intercept_
# %%
