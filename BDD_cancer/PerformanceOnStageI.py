import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve
# from treeple import HonestForestClassifier
# from treeple.tree import MultiViewDecisionTreeClassifier
# from treeple.stats import build_oob_forest

pd.set_option('future.no_silent_downcasting', True)

# Data Loading

## Read csv file
data = pd.read_csv("./data/WiseCondorX.Wise-1.csv")
## Get label
data['Cancer Status'] = data['Cancer Status'].replace({"Healthy": 0, "Cancer": 1}).astype(int)
## Drop columns with all NaN values
data = data.dropna(axis=1, how='all')
data = data.fillna(0)
        
if 'Stage' in data.columns:
    stage_mapping = {
        'IA': 'I', 'IA1': 'I', 'IA2': 'I', 'IA3': 'I', 'IB': 'I', 'IC': 'I',
        'IIA': 'II', 'IIB': 'II', 'IIC': 'II', 'II': 'II',
        'IIIA': 'III', 'IIIA1': 'III', 'IIIA2': 'III', 'IIIIA': 'III', 'IIIB': 'III', 'IIIC': 'III', 'III': 'III', 'pT3N2Mx': 'III',
        'IVA': 'IV', 'IVB': 'IV', 'IV': 'IV',
        'Normal': 'Normal',
        'Not given': 'Unknown',
        'nan': 'Unknown',
        'biopsy only at time of blood draw': 'Unknown',
        '0': '0',
        '0__TisN0M0': '0',
        '0___TisN0M0': '0',
        '0__Tis(2)N0M0': '0',
        'IIIV': 'Unknown'  # Maybe wrong
    }
    data['Stage'] = data['Stage'].astype(str).map(stage_mapping).fillna('Unknown')
      
## Drop columns with non-numeric values
non_feature_cols = ['Run', 'Library', 'Tumor type', 'Library volume (uL)', 
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
for col in non_feature_cols:
    if col in data.columns:
        data = data.drop(columns=[col])
        
## Data split
X_test = data[data['Stage'] == 'I'].drop(columns=['Cancer Status', 'Stage']) 
y_test = data[data['Stage'] == 'I']['Cancer Status'] 
X_train = data[data['Stage'] != 'I'].drop(columns=['Cancer Status', 'Stage']) 
y_train = data[data['Stage'] != 'I']['Cancer Status']

# Save data
splitW1_dir = './data/splitedW1'
if not os.path.exists(splitW1_dir):
    os.makedirs(splitW1_dir) 
X_test.to_csv(os.path.join(splitW1_dir, 'X_test_I.csv'), index=False)
y_test.to_csv(os.path.join(splitW1_dir, 'y_test_I.csv'), index=False)
X_train.to_csv(os.path.join(splitW1_dir, 'X_train_expI.csv'), index=False)
y_train.to_csv(os.path.join(splitW1_dir, 'y_train_expI.csv'), index=False)

print(f"Training Set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing Set: X_test={X_test.shape}, y_test={y_test.shape}")




