import os
from treeple import HonestForestClassifier, ObliqueRandomForestClassifier
from treeple.tree import MultiViewDecisionTreeClassifier, ObliqueDecisionTreeClassifier

# **Define Model Configurations**
MODEL_CONFIGS = {
    "might": {
        "class": HonestForestClassifier,
        "params": {
            "n_estimators": 5000,  
            "honest_fraction": 0.5,
            "n_jobs": 8,  
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
            "n_jobs": 8,
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
            "n_jobs": 8,
            "bootstrap": True,
            "max_features": 0.3,
        }
    }
}

def train_model(model_name, X_train, y_train, timestamp):
    """ Train a model and return the trained model object """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Training {model_name} model...")

    model_class = MODEL_CONFIGS[model_name]["class"]
    model_params = MODEL_CONFIGS[model_name]["params"]
    model = model_class(**model_params)

    # **Train model**
    model.fit(X_train, y_train)

    print(f"{model_name} training completed.")

    return model  # **Return trained model**