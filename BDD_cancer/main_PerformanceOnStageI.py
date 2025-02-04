import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from models.model_manager import train_model, evaluate_model

pd.set_option('future.no_silent_downcasting', True)

#######################################################################################
# Load Data

## Read Data
data = pd.read_csv("./data/WiseCondorX.Wise-1.csv")

## Work with Data
data['Cancer Status'] = data['Cancer Status'].replace({"Healthy": 0, "Cancer": 1}).astype(int)
data = data.dropna(axis=1, how='all').fillna(0)
if 'Stage' in data.columns:
    stage_mapping = {
        'IA': 'I', 'IA1': 'I', 'IA2': 'I', 'IA3': 'I', 'IB': 'I', 'IC': 'I',
        'IIA': 'II', 'IIB': 'II', 'IIC': 'II', 'II': 'II',
        'IIIA': 'III', 'IIIA1': 'III', 'IIIA2': 'III', 'IIIIA': 'III',
        'IIIB': 'III', 'IIIC': 'III', 'III': 'III', 'pT3N2Mx': 'III',
        'IVA': 'IV', 'IVB': 'IV', 'IV': 'IV',
        'Normal': 'Normal', 'Not given': 'Unknown',
        'nan': 'Unknown', 'biopsy only at time of blood draw': 'Unknown',
        '0': '0', '0__TisN0M0': '0', '0___TisN0M0': '0', '0__Tis(2)N0M0': '0',
        'IIIV': 'Unknown'
    }
    data['Stage'] = data['Stage'].astype(str).map(stage_mapping).fillna('Unknown')

non_feature_cols = ['Run', 'Library', 'Tumor type', 'Library volume (uL)',
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
for col in non_feature_cols:
    if col in data.columns:
        data = data.drop(columns=[col])

#######################################################################################
# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

stage_I_samples = data[data['Stage'] == 'I']
normal_samples = data[data['Stage'] == 'Normal']

def run_fold(train_idx, test_idx, fold):
    """ Run a single fold of 5-fold cross validation """
    print(f"\n Running Fold {fold+1}/5 ...")

    # Divide data into train and test
    normal_test = normal_samples.iloc[test_idx]  # 20% Normal
    test_samples = pd.concat([stage_I_samples, normal_test])  # Test = 20% Stage I + 20% Normal
    train_samples = pd.concat([data, test_samples]).drop_duplicates(keep=False)  # Train = All - Test

    X_train, y_train = train_samples.drop(columns=['Cancer Status', 'Stage', 'Sample']), train_samples['Cancer Status'].values.ravel()
    X_test, y_test = test_samples.drop(columns=['Cancer Status', 'Stage', 'Sample']), test_samples['Cancer Status'].values.ravel()

    print(f"Fold {fold+1} -> Train: {X_train.shape}, Test: {X_test.shape}")

    # Parallel training
    Parallel(n_jobs=3)(
        delayed(train_model)(model, X_train, y_train) for model in ["might", "SPO-MIGHT", "SPORF"]
    )

    # Parallel evaluation
    fold_results = Parallel(n_jobs=3)(
        delayed(evaluate_model)(model, X_test, y_test) for model in ["might", "SPO-MIGHT", "SPORF"]
    )

    return fold_results

# Run 5-Fold Cross Validation
results = Parallel(n_jobs=5)(
    delayed(run_fold)(train_idx, test_idx, fold) for fold, (train_idx, test_idx) in enumerate(kf.split(normal_samples))
)

#######################################################################################
# Display Results
print("\n Final Results Across 5 Folds:")
for fold_id, fold_res in enumerate(results):
    print(f"Fold {fold_id+1}: {fold_res}")


# # For Single Fold
# train_idx, test_idx = next(iter(kf.split(normal_samples)))

# # 
# single_fold_results = run_fold(train_idx, test_idx, fold=0)

# print("\nResults for Single Fold:")
# print(single_fold_results)