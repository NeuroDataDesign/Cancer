import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from treeple.stats import build_oob_forest

# Read the test data
data_dir = "/home/sunyvxuan/projects/Cancer/BDD_cancer/data/splitedW1"
X_test = pd.read_csv(os.path.join(data_dir, "X_test_I.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test_I.csv")).values.ravel()

# Model directory
model_dir = "/home/sunyvxuan/projects/Cancer/BDD_cancer/models/trained"

def load_model(model_name):
    """Load the specified model from disk."""
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_name}, please run train_models.py first.")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)

def test_model(model_name):
    """Test the specified model and plot the ROC curve."""
    print(f"Testing {model_name} model...")
    model = load_model(model_name)
    
    # **不同模型的预测方式**
    if model_name in ["might", "SPO-MIGHT", "SPORF"]:
        print("Using OOB predictions...")
        _, posterior_arr = build_oob_forest(model, X_test, y_test, verbose=False)
        posterior_mean = np.nanmean(posterior_arr, axis=0)  # Shape: (n_samples, n_classes)
    else:
        print("Using direct prediction...")
        posterior_mean = model.predict_proba(X_test)  # Shape: (n_samples, 2)
    
    # **确保取 `class=1` 的概率**
    if posterior_mean.ndim == 2 and posterior_mean.shape[1] == 2:
        posterior_final = posterior_mean[:, 1]  # 取 class=1 概率
    else:
        print(f"Warning: Unexpected prediction shape {posterior_mean.shape}. Using last column.")
        posterior_final = posterior_mean.ravel()  # Flatten if needed

    # **处理 NaN**
    if np.isnan(posterior_final).any():
        print("Warning: posterior_final contains NaN values!")
        posterior_final = np.nan_to_num(posterior_final, nan=np.nanmedian(posterior_final))

    # **计算 ROC 曲线**
    fpr, tpr, _ = roc_curve(y_test, posterior_final, pos_label=1)
    auc_score = roc_auc_score(y_test, posterior_final)
    
    # **计算 S@98**
    S98 = np.max(tpr[fpr <= 0.02]) if np.any(fpr <= 0.02) else 0

    # **绘制 ROC 曲线**
    fig_dir = "/home/sunyvxuan/projects/Cancer/BDD_cancer/figures"
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # **标记 S@98**
    if np.any(fpr <= 0.02):
        fpr_S98 = fpr[fpr <= 0.02][-1]  # 最后一个满足 FPR <= 0.02 的值
        tpr_S98 = S98
        plt.scatter(fpr_S98, tpr_S98, color='red', label=f'S@98: {S98:.3f}', zorder=3)

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{model_name} Model ROC Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{fig_dir}/{model_name}_ROC.png")
    plt.close()
    print(f"ROC curve saved to {fig_dir}/{model_name}_ROC.png")

    return S98
    
if __name__ == "__main__":
    test_model("might")  