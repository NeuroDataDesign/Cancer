import os
import pickle
import datetime
from treeple import HonestForestClassifier, ObliqueRandomForestClassifier
from treeple.tree import MultiViewDecisionTreeClassifier, ObliqueDecisionTreeClassifier

# Make sure the models/trained directory exists
os.makedirs("models/trained", exist_ok=True)
model_dir = "models/trained"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

##########################################################################
# Define Model Configurations
MODEL_CONFIGS = {
    "might": {
        "class": HonestForestClassifier,
        "params": {
            "n_estimators": 5000,
            "honest_fraction": 0.5,
            "n_jobs": 40,
            "bootstrap": True,
            "stratify": True,
            "max_samples": 1.0,
            "max_features": 0.3,
            "tree_estimator": MultiViewDecisionTreeClassifier(),
        }
    },
    "SPO-MIGHT": {
        "class": HonestForestClassifier,
        "params": {
            "n_estimators": 5000,
            "honest_fraction": 0.5,
            "n_jobs": 40,
            "bootstrap": True,
            "stratify": True,
            "max_samples": 1.0,
            "max_features": 0.3,
            "tree_estimator": ObliqueDecisionTreeClassifier(),
        }
    },
    "SPORF": {
        "class": ObliqueRandomForestClassifier,
        "params": {
            "n_estimators": 5000,
            "n_jobs": 40,
            "bootstrap": True,
            "max_features": 0.3,
        }
    }
}

##########################################################################
# Train and Save Model
def train_and_save_model(model_name, X_train, y_train):
    """Train and save a model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"🔹 Training {model_name} Model...")
    
    # Instantiate Model
    model_class = MODEL_CONFIGS[model_name]["class"]
    model_params = MODEL_CONFIGS[model_name]["params"]
    model = model_class(**model_params)

    # Fit Model
    model.fit(X_train, y_train)

    # Save Model
    model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ {model_name} Model has been trained and saved to {model_path}")