import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def ce_loss(theta, intercept, feature_matrix, labels):
    z = np.multiply(labels, (np.dot(feature_matrix, theta)) + intercept)
    return np.sum(np.log(np.exp(-z) + 1.0)) / feature_matrix.shape[0]

def get_auc(theta, intercept, feature_matrix, labels):
    scores = np.dot(feature_matrix, theta) + intercept
    # scores = np.exp(-z)
    fpr, tpr, _ = roc_curve(y_true = labels, y_score = scores)
    auc = roc_auc_score(y_true = labels, y_score = scores)
    return auc, (tpr, fpr)

def plot_auc(tpr, fpr, auc, title, time_str):
    plt.figure(figsize = (7,5))
    plt.plot(fpr, tpr, label = f"AUC: {auc}")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.savefig(f"./out/{time_str}-{title}.pdf")