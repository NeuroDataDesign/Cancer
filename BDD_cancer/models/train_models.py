import os
from treeple import HonestForestClassifier, ObliqueRandomForestClassifier
from treeple.tree import MultiViewDecisionTreeClassifier, ObliqueDecisionTreeClassifier

# **Define Model Configurations**

# Hyperparameters
n_estimators = 50000
max_features = 'sqrt'
n_jobs = 40
honest_fraction = 0.367
max_samples = 1.6

MODEL_CONFIGS = {
    "MIGHT": {
        "class": HonestForestClassifier,
        "params": {
            "n_estimators": n_estimators,  
            "honest_fraction": honest_fraction,
            "n_jobs": n_jobs,  
            "bootstrap": True,
            "stratify": True,
            "max_samples": max_samples,
            "max_features": max_features,
            "tree_estimator": MultiViewDecisionTreeClassifier(),
        }
    },
    "SPO-MIGHT": {
        "class": HonestForestClassifier,
        "params": {
            "n_estimators": n_estimators,
            "honest_fraction": honest_fraction,
            "n_jobs": n_jobs,
            "bootstrap": True,
            "stratify": True,
            "max_samples": max_samples,
            "max_features": max_features,
            "tree_estimator": ObliqueDecisionTreeClassifier(),
        }
    },
    "SPORF": {
        "class": ObliqueRandomForestClassifier,
        "params": {
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "bootstrap": True,
            "max_features": max_features,
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
    # model_params["bootstrap"] = True
    model = model_class(**model_params)

    # **Train model**
    model.fit(X_train, y_train)

    print(f"{model_name} training completed.")

    return model  # **Return trained model**