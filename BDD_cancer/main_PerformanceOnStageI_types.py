import os
import numpy as np
import pandas as pd
import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from models.train_models import train_model
from models.test_models import test_model

pd.set_option('future.no_silent_downcasting', True)

#######################################################################################
#  Load Data

# **Read CSV file**
data = pd.read_csv("./data/WiseCondorX.Wise-1.csv")

# **Process Cancer Status**
data['Cancer Status'] = data['Cancer Status'].replace({"Healthy": 0, "Cancer": 1}).astype(int)

# **Handle missing values**
data = data.dropna(axis=1, how='all').fillna(0)

# **Normalize Stage**
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

# **Remove non-feature columns**
non_feature_cols = ['Run', 'Library', 'Library volume (uL)', 'Library Volume',
                    'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
data = data.drop(columns=[col for col in non_feature_cols if col in data.columns])

#######################################################################################
#  Extract Stage I Samples and Group by Tumor Type
stage_I_samples = data[data['Stage'] == 'I']
tumor_types = stage_I_samples['Tumor type'].unique()

# **Store results for S@98 for each Tumor Type**
tumor_type_results = {}

#######################################################################################
# Perform 5-Fold Cross Validation for Each Tumor Type
kf = KFold(n_splits=5, shuffle=True, random_state=42)

normal_samples = data[data['Stage'] == 'Normal']

def run_fold(train_idx, test_idx, fold, tumor_type):
    """ Train and test models in parallel within a single fold for a specific Tumor Type """
    print(f"\n Running Fold {fold+1}/5 for Tumor Type: {tumor_type}...")

    # **Select samples for the specific tumor type**
    stage_I_tumor_samples = stage_I_samples[stage_I_samples['Tumor type'] == tumor_type]
    
    # **Split train & test data**
    normal_test = normal_samples.iloc[test_idx]  # Select 20% Normal samples
    test_samples = pd.concat([stage_I_tumor_samples, normal_test])  # Test = Specific Tumor Type + 20% Normal
    train_samples = pd.concat([data, test_samples]).drop_duplicates(keep=False)  # Train = All - Test

    X_train, y_train = train_samples.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type']), train_samples['Cancer Status'].values.ravel()
    X_test, y_test = test_samples.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type']), test_samples['Cancer Status'].values.ravel()

    print(f"Tumor Type {tumor_type} - Fold {fold+1} -> Train: {X_train.shape}, Test: {X_test.shape}")

    # **Generate a single timestamp for this fold**
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # **Train models in parallel**
    trained_models = Parallel(n_jobs=3)(
        delayed(train_model)(model, X_train, y_train, timestamp) 
        for model in ["might", "SPO-MIGHT", "SPORF"]
    )
    
    # **Map model names to trained model objects**
    trained_models_dict = {model_name: model for model_name, model in zip(["might", "SPO-MIGHT", "SPORF"], trained_models)}

    # Parallel test models
    fold_results = Parallel(n_jobs=3)(
        delayed(test_model)(trained_models_dict[model], model, X_test, y_test, timestamp)
        for model in trained_models_dict
    )

    return fold_results

# **Loop through each Tumor Type**
for tumor_type in tumor_types:
    print(f"\n### Processing Tumor Type: {tumor_type} ###")
    
    tumor_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(normal_samples)):
        fold_result = run_fold(train_idx, test_idx, fold, tumor_type)
        tumor_results.append(fold_result)  # Store results for each fold

    tumor_type_results[tumor_type] = tumor_results  # Store results per tumor type

#######################################################################################
#  Print Final S@98 Results for Each Tumor Type
print("\n Final S@98 Results Across Tumor Types:")
for tumor_type, fold_res in tumor_type_results.items():
    print(f"Tumor Type: {tumor_type}")
    for fold_id, fold_s98 in enumerate(fold_res):
        print(f"  Fold {fold_id+1}: {fold_s98}")