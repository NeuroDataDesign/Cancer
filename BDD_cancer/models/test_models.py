import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from treeple.stats import build_oob_forest

# make sure `figures` directory exists
os.makedirs("./figures", exist_ok=True)
fig_dir = "./figures"
model_dir = "models/trained"
latest_model_file = os.path.join(model_dir, "latest_model.txt")

def load_latest_model(model_name):
    """Load the latest model for a given model name"""
    if not os.path.exists(latest_model_file):
        raise FileNotFoundError("Cannot find latest_model.txt. Please run train_models.py first.")

    with open(latest_model_file, "r") as f:
        model_paths = f.readlines()

    # Read model paths from latest_model.txt
    model_paths = [line.strip().split(": ") for line in model_paths if ": " in line]
    model_dict = {m[0]: m[1] for m in model_paths}

    if model_name not in model_dict:
        raise FileNotFoundError(f"Latest model for {model_name} not found. Ensure train_models.py has run.")

    latest_model_path = model_dict[model_name]

    if not os.path.exists(latest_model_path):
        raise FileNotFoundError(f"Latest model file not found: {latest_model_path}")

    print(f"🔵 Loading {model_name} model from {latest_model_path}")
    
    # Extract timestamp from model path
    timestamp = latest_model_path.split("_")[-1].split(".")[0]

    # Load model
    with open(latest_model_path, "rb") as f:
        model = pickle.load(f)

    return model, timestamp

def test_model(model_name, X_test, y_test):
    """Test a model and plot ROC curve"""
    print(f"🔹 Testing {model_name} model...")
    model, timestamp = load_latest_model(model_name)

    # Use OOB predictions for MIGHT, SPO-MIGHT, and SPORF
    if model_name in ["might", "SPO-MIGHT", "SPORF"]:
        print("Using OOB predictions...")
        _, posterior_arr = build_oob_forest(model, X_test, y_test, verbose=False)
        posterior_mean = np.nanmean(posterior_arr, axis=0)  # (n_samples, n_classes)
    else:
        print("Using direct prediction...")
        posterior_mean = model.predict_proba(X_test)  # (n_samples, 2)
    
    # make sure posterior_mean is 1D
    if posterior_mean.ndim == 2 and posterior_mean.shape[1] == 2:
        posterior_final = posterior_mean[:, 1]
    else:
        posterior_final = posterior_mean.ravel()

    # Replace NaNs with median
    if np.isnan(posterior_final).any():
        posterior_final = np.nan_to_num(posterior_final, nan=np.nanmedian(posterior_final))

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, posterior_final, pos_label=1)
    auc_score = roc_auc_score(y_test, posterior_final)
    
    # Compute S@98
    S98 = np.max(tpr[fpr <= 0.02]) if np.any(fpr <= 0.02) else 0

    # Save ROC curve
    figure_path = f"{fig_dir}/{model_name}_ROC_{timestamp}.png"

    # Plot ROC curve
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # Highlight S@98
    if np.any(fpr <= 0.02):
        plt.scatter(fpr[fpr <= 0.02][-1], S98, color='red', label=f'S@98: {S98:.3f}', zorder=3)

    plt.xlabel('False Positive Rate (FPR)', fontsize=20)
    plt.ylabel('True Positive Rate (TPR)', fontsize=20)
    plt.title(f'{model_name} Model ROC Curve', fontsize=24)
    plt.legend()
    plt.grid(alpha=0.5)

    # Save figure
    plt.savefig(figure_path)
    plt.close()

    print(f"✅ ROC curve saved to {figure_path}")

    return S98

# if __name__ == "__main__":
#     test_model("might", X_test, y_test)