import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from treeple.stats import build_oob_forest

# Read the test data
data_dir = "./data/splitedW1"
X_test = pd.read_csv(os.path.join(data_dir, "X_test_I.csv"))
y_test = pd.read_csv(os.path.join(data_dir, "y_test_I.csv")).values.ravel()

# Model directory
model_dir = "./models/trained"

def load_model(model_name):
    """Load the specified model from disk."""
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Can not find {model_name}, please run train_models.py fisrt.")
    
    with open(model_path, "rb") as f:
        return pickle.load(f)

def test_model(model_name):
    """Test the specified model and plot the ROC curve."""
    print(f"Testing {model_name} model...")
    model = load_model(model_name)
    
    _, posterior_arr = build_oob_forest(model, X_test, y_test, verbose=False)
    fpr, tpr, _ = roc_curve(y_test, posterior_arr[:, -1], pos_label=1)
    auc_score = roc_auc_score(y_test, posterior_arr[:, -1])

    # Draw ROC curve and save it
    fig_dir = "/home/sunyvxuan/projects/Cancer/BDD_cancer/figures"
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{model_name} Model ROC Curve')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{fig_dir}/{model_name}_ROC.png")
    plt.close()
    print(f"ROC curve saved to {fig_dir}/{model_name}_ROC.png")
    
if __name__ == "__main__":
    test_model("might")  