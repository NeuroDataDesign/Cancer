import os
import numpy as np
import pandas as pd
import datetime
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from models.train_models import train_model
from models.test_models import test_model
import gc
from joblib.externals.loky import get_reusable_executor

pd.set_option('future.no_silent_downcasting', True)

#######################################################################################
# **1️⃣ Load Data**

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
non_feature_cols = ['Run', 'Library', 'Tumor type', 'Library volume (uL)',
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
data = data.drop(columns=[col for col in non_feature_cols if col in data.columns])

#######################################################################################
# **2️⃣ 5-Fold Cross Validation**
kf = KFold(n_splits=5, shuffle=True, random_state=210)

stage_I_samples = data[data['Stage'] == 'I']
stage_I_test = stage_I_samples.sample(n=100, random_state=425)  
# stage_I_train = stage_I_samples.drop(stage_I_test.index)
normal_samples = data[data['Stage'] == 'Normal']

def run_fold(train_idx, test_idx, fold):
    """ Train and test models in parallel within a single fold """
    print(f"\n Running Fold {fold+1}/5 ...")

    # **Split train & test data**
    normal_test = normal_samples.iloc[test_idx]  # Select 20% Normal samples
    test_samples = pd.concat([stage_I_test, normal_test])  # Test = Stage I + 20% Normal
    train_samples = pd.concat([data, test_samples]).drop_duplicates(keep=False)  # Train = All - Test

    X_train, y_train = train_samples.drop(columns=['Cancer Status', 'Stage', 'Sample']), train_samples['Cancer Status'].values.ravel()
    X_test, y_test = test_samples.drop(columns=['Cancer Status', 'Stage', 'Sample']), test_samples['Cancer Status'].values.ravel()

    print(f"Fold {fold+1} -> Train: {X_train.shape}, Test: {X_test.shape}")

    # **Generate a single timestamp for this fold**
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # **Train models in parallel**
    trained_models = Parallel(n_jobs=3)(
        delayed(train_model)(model, X_train, y_train, timestamp) 
        for model in ["might", "SPO-MIGHT", "SPORF"]
    )
    
    print("Clearing joblib processes and garbage collection...")
    get_reusable_executor().shutdown(wait=True)
    gc.collect()
    print("Processes cleared. Proceeding...")
    
    # Map model names to trained model objects
    print("Creating trained_models_dict...")
    trained_models_dict = {model_name: model for model_name, model in zip(["might", "SPO-MIGHT", "SPORF"], trained_models)}
    print("trained_models_dict created successfully.")
    
        # Parallel test models
    fold_results = Parallel(n_jobs=3)(
        delayed(test_model)(trained_models_dict[model], model, X_test, y_test, timestamp)
        for model in trained_models_dict
    )
    return fold_results

# **3️⃣ Run Each Fold Sequentially (Train and Test are Parallel)**
results = []
for fold, (train_idx, test_idx) in enumerate(kf.split(normal_samples)):
    fold_result = run_fold(train_idx, test_idx, fold)
    results.append(fold_result)  # Store results for each fold

#######################################################################################
# **4️⃣ Print Final Results**
print("\n Final Results Across 5 Folds:")
for fold_id, fold_res in enumerate(results):
    print(f"Fold {fold_id+1}: {fold_res}")