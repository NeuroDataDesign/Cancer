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

random_state = 42


# Load Data and Process Cancer Status
data = pd.read_csv("./data/WiseCondorX.Wise-1.csv")
data['Cancer Status'] = data['Cancer Status'].replace({"Healthy": 0, "Cancer": 1}).astype(int)

# Handle missing values
data = data.dropna(axis=1, how='all').fillna(0)

# Stage Normalization
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

# Remove non-feature columns
non_feature_cols = ['Run', 'Library', 'Library volume (uL)',
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
data = data.drop(columns=[col for col in non_feature_cols if col in data.columns])

# Classify samples based on stage and cancer type
cancer_types_test = {'Stomach', 'Pancreas', 'Esophagus', 'Lung', 'Liver', 'Ovary'}
test_set = data[(data['Stage'] == 'I') & (data['Tumor type'].isin(cancer_types_test))]
normal_samples = data[data['Stage'] == 'Normal']
normal_test_samples = normal_samples.sample(n=256, random_state=random_state)
test_set = pd.concat([test_set, normal_test_samples])
train_set = data.drop(test_set.index)
# print(test_set['Tumor type'].value_counts())
# print(train_set['Tumor type'].value_counts())

# X_train, y_train = train_set.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type']), train_set['Cancer Status'].values.ravel()
X_test, y_test = test_set.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type']), test_set['Cancer Status'].values.ravel()
# print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
total_train_samples = len(train_set)

train_sizes = []
size = 64
MODEL_NAMES = ["MIGHT", "SPO-MIGHT", "SPORF"]


while size < total_train_samples:
    train_sizes.append(size)
    size *= 2
if train_sizes[-1] != total_train_samples:
    train_sizes.append(total_train_samples)
    
    
def train_and_test(model_name, size):
    """Train and test a model with a specific dataset size."""
    print(f"\nTraining {model_name} with {size} samples...")

    # Calculate the proportion of cancer and healthy samples in the training set
    total_cancer = train_set[train_set['Cancer Status'] == 1].shape[0]
    total_healthy = train_set[train_set['Cancer Status'] == 0].shape[0]
    total_samples = total_cancer + total_healthy

    cancer_ratio = total_cancer / total_samples 
    healthy_ratio = total_healthy / total_samples  

    num_cancer = int(size * cancer_ratio)
    num_healthy = size - num_cancer  

    # Ensure at least one sample of each class
    num_cancer = min(num_cancer, total_cancer)
    num_healthy = min(num_healthy, total_healthy)

    # Sample the training set
    train_cancer_samples = train_set[train_set['Cancer Status'] == 1].sample(n=num_cancer, random_state=random_state)
    train_healthy_samples = train_set[train_set['Cancer Status'] == 0].sample(n=num_healthy, random_state=random_state)

    # Combine the samples and shuffle the dataset
    train_subset = pd.concat([train_cancer_samples, train_healthy_samples]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_train_subset = train_subset.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type'])
    y_train_subset = train_subset['Cancer Status'].values.ravel()
    
    model = train_model(model_name, X_train_subset, y_train_subset, timestamp=None)

    S98 = test_model(model, model_name, X_test, y_test, timestamp=None)

    return model_name, size, S98

# Run training & testing in parallel for all models and dataset sizes
results = Parallel(n_jobs=40)(
    delayed(train_and_test)(model_name, size)
    for model_name in MODEL_NAMES
    for size in train_sizes
)

gc.collect()

print("\nFinal Results:")
for model_name, size, S98 in results:
    print(f"Model: {model_name}, Train Size: {size}, S@98: {S98:.4f}")