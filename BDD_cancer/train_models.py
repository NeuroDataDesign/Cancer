import os
import pandas as pd
import pickle
from treeple import HonestForestClassifier
from treeple.tree import MultiViewDecisionTreeClassifier
from treeple.stats import build_oob_forest

# Make sure the models directory exists
os.makedirs("models", exist_ok=True)

# Load the data
splitW1_dir = './data/splitedW1'
X_train = pd.read_csv(os.path.join(splitW1_dir, 'X_train_expI.csv'))
y_train = pd.read_csv(os.path.join(splitW1_dir, 'y_train_expI.csv')).values.ravel()  # 确保 y_train 变成 1D 数组

# Define the MIGHT model hyperparameters
might_kwargs = {
    "n_estimators": 10000,
    "honest_fraction": 0.7,
    "n_jobs": 40,
    "bootstrap": True,
    "stratify": True,
    "max_samples": 1.0,
    "max_features": 0.2,
    "tree_estimator": MultiViewDecisionTreeClassifier(),
}

# Train the MIGHT model
print("Training MIGHT model...")
might_model = HonestForestClassifier(**might_kwargs)
might_model.fit(X_train, y_train)

# Save the MIGHT model
might_model_path = "./models/might_model.pkl"
with open(might_model_path, "wb") as f:
    pickle.dump(might_model, f)
print(f"MIGHT Model has been saved to {might_model_path}")

# Other models can be trained in a similar way