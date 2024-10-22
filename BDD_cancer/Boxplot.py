from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Obtain the list of samples
sample_list_file = "AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]

# Obtain the list of cancer samples
Cancer_samples = sample_list[sample_list["cohort"] != "PanelOfNormals"]["sample_id"]
print(len(Cancer_samples))

# Read the data
data = pd.read_csv("WiseCondorX.Wise-1.csv", low_memory=False)

# unique_stages = data['Stage'].unique()
# print("Unique cancer stages:", unique_stages)
# stage_counts = data['Stage'].value_counts(dropna=False) 
# print("Cancer stage counts:\n", stage_counts)

# Filter the data to include only the cancer samples
stage_iv = data[data['Stage'].isin(['IV', 'IVA', 'IVB'])]
stage_i = data[data['Stage'].isin(['I', 'IA', 'IB', 'IC', 'IA1', 'IA2', 'IA3'])]

# print(data.columns)

# Filter the data to include only the cancer samples
Cancer_data = data[data['Sample'].isin(Cancer_samples)]

# Drop columns
non_numeric_cols = ['Run', 'Sample','Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
Cancer_data_numeric = Cancer_data.drop(columns=non_numeric_cols, errors='ignore')
nan_cols = Cancer_data_numeric.isnull().all(axis=0).to_numpy()
Cancer_data_numeric = Cancer_data_numeric.loc[:, ~nan_cols]
Cancer_data_numeric = Cancer_data_numeric.infer_objects()
Cancer_data_numeric = Cancer_data_numeric.apply(pd.to_numeric, errors='coerce')
Cancer_data_numeric.fillna(0, inplace=True)

# Set the index to the Sample column
Cancer_data_numeric.set_index(Cancer_data['Sample'], inplace=True)

# print(Cancer_data_numeric.describe())

# Plot the boxplot
# plt.figure(figsize=(20, 8))
# sns.boxplot(data=Cancer_data_numeric, orient='h', palette="Set3")
# plt.title('Boxplot of Genomic Regions Data Distribution')
# plt.xlabel('Genomic Regions Values')
# plt.ylabel('Genomic Regions')
# plt.show()


stage_iv_numeric = stage_iv.drop(columns=non_numeric_cols, errors='ignore')
stage_i_numeric = stage_i.drop(columns=non_numeric_cols, errors='ignore')


stage_iv_numeric = stage_iv_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
stage_i_numeric = stage_i_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)


plt.figure(figsize=(15, 6))
sns.boxplot(data=stage_iv_numeric, orient='h', palette="coolwarm")
plt.title('Boxplot of Genomic Regions for Stage IV')
plt.xlabel('Genomic Regions Values')
plt.ylabel('Genomic Regions')
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(data=stage_i_numeric, orient='h', palette="coolwarm")
plt.title('Boxplot of Genomic Regions for Stage I')
plt.xlabel('Genomic Regions Values')
plt.ylabel('Genomic Regions')
plt.show()