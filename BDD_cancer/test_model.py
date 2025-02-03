import pickle
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from treeple.stats import build_oob_forest

# Load the test data
splitW1_dir = './data/splitedW1'
X_test = pd.read_csv(os.path.join(splitW1_dir, 'X_test_I.csv'))
y_test = pd.read_csv(os.path.join(splitW1_dir, 'y_test_I.csv')).values.ravel()

# Make sure the MIGHT model exists
if not os.path.exists("models/might_model.pkl"):
    raise FileNotFoundError("MIGHT model not found. Please train the model first.")

# Load the MIGHT model
with open("./models/might_model.pkl", "rb") as f:
    might_model = pickle.load(f)

print("MIGHT model loaded successfully.")

# Use the MIGHT model to predict the test data
_, posterior_arr = build_oob_forest(might_model, X_test, y_test, verbose=False)
fpr, tpr, _ = roc_curve(y_test, posterior_arr[:, -1], pos_label=1)
auc_score = roc_auc_score(y_test, posterior_arr[:, -1])

# Draw the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"MIGHT (AUC={auc_score:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('MIGHT Model ROC Curve')
plt.legend()
plt.grid(alpha=0.5)
plt.show()