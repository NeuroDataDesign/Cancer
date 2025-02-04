import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from treeple.stats import build_oob_forest

# Make sure the figures directory exists
os.makedirs("./figures", exist_ok=True)
fig_dir = "./figures"

# Get current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Load the trained model
def load_model(model_name):
    """Load a trained model"""
    model_path = os.path.join("./models/trained", f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find {model_name}, please run train_models.py first.")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Evaluate the model
def test_model(model_name, X_test, y_test):
    """Test a trained model"""
    print(f"🔹 Testing {model_name} model...")
    model = load_model(model_name)
    
    # Use OOB predictions for MIGHT, SPO-MIGHT, and SPORF
    if model_name in ["might", "SPO-MIGHT", "SPORF"]:
        print("Using OOB predictions...")
        _, posterior_arr = build_oob_forest(model, X_test, y_test, verbose=False)
        posterior_mean = np.nanmean(posterior_arr, axis=0)  # Shape: (n_samples, n_classes)
    else:
        print("Using direct prediction...")
        posterior_mean = model.predict_proba(X_test)  # Shape: (n_samples, 2)
    
    # Make sure the posterior_mean is 1D
    if posterior_mean.ndim == 2 and posterior_mean.shape[1] == 2:
        posterior_final = posterior_mean[:, 1]
    else:
        posterior_final = posterior_mean.ravel()

    # Handle NaN values
    if np.isnan(posterior_final).any():
        posterior_final = np.nan_to_num(posterior_final, nan=np.nanmedian(posterior_final))

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, posterior_final, pos_label=1)
    auc_score = roc_auc_score(y_test, posterior_final)
    
    # Compute S@98
    S98 = np.max(tpr[fpr <= 0.02]) if np.any(fpr <= 0.02) else 0

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # Highlight S@98
    if np.any(fpr <= 0.02):
        plt.scatter(fpr[fpr <= 0.02][-1], S98, color='red', label=f'S@98: {S98:.3f}', zorder=3)

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{model_name} Model ROC Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{fig_dir}/{model_name}_ROC_{timestamp}.png")
    plt.close()
    print(f"✅ ROC curve saved to {fig_dir}/{model_name}_ROC.png")

    return S98