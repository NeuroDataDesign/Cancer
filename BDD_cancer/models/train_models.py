import os
import pandas as pd
import pickle
from treeple import HonestForestClassifier
from treeple import ObliqueRandomForestClassifier
from treeple.tree import MultiViewDecisionTreeClassifier
from treeple.tree import ObliqueDecisionTreeClassifier

# Make sure the models directory exists
os.makedirs("models", exist_ok=True)

##########################################################################
# Load the data
splitW1_dir = './data/splitedW1'
X_train = pd.read_csv(os.path.join(splitW1_dir, 'X_train_expI.csv'))
y_train = pd.read_csv(os.path.join(splitW1_dir, 'y_train_expI.csv')).values.ravel()  # 确保 y_train 变成 1D 数组


##########################################################################
# Define the MIGHT model hyperparameters
MODELS = {
    "might": HonestForestClassifier(
            n_estimators=5000,
            honest_fraction=0.5,
            n_jobs=40,
            bootstrap=True,
            stratify=True,
            max_samples=1.0,
            max_features=0.3,
            tree_estimator=MultiViewDecisionTreeClassifier(),
        ),
    "SPO-MIGHT": HonestForestClassifier(
            n_estimators=5000,
            honest_fraction=0.5,
            n_jobs=40,
            bootstrap=True,
            stratify=True,
            max_samples=1.0,
            max_features=0.3,
            tree_estimator=ObliqueDecisionTreeClassifier(),
        ),
    "SPORF": ObliqueRandomForestClassifier(
            n_estimators=5000,
            n_jobs=40,
            bootstrap=True,
            max_features=0.3,
        ),
        
}

model_dir = "models/trained"

def train_and_save_model(model_name):
    """Train the specified model and save it to disk."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Training {model_name} Model...")
    model = MODELS[model_name]
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"{model_name} Model has been trained, save to {model_path}")

if __name__ == "__main__":
    # Train and save all models
    for model_name in MODELS.keys():
        train_and_save_model(model_name)