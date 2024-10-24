from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Obtain the list of samples
sample_list_file = "AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]

# Obtain the list of cancer samples
Cancer_samples = sample_list[sample_list["cohort"] != "PanelOfNormals"]["sample_id"]
Normal_samples = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]
print(len(Cancer_samples))
print(len(Normal_samples))

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
Nornal_data = data[data['Sample'].isin(Normal_samples)]

# Drop columns
non_numeric_cols = ['Run', 'Sample','Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
Cancer_data_numeric = Cancer_data.drop(columns=non_numeric_cols, errors='ignore')
Normal_data_numeric = Nornal_data.drop(columns=non_numeric_cols, errors='ignore')
nan_cols = Cancer_data_numeric.isnull().all(axis=0).to_numpy()
nan_cols_normal = Normal_data_numeric.isnull().all(axis=0).to_numpy()
Cancer_data_numeric = Cancer_data_numeric.loc[:, ~nan_cols]
Normal_data_numeric = Normal_data_numeric.loc[:, ~nan_cols_normal]
Cancer_data_numeric = Cancer_data_numeric.infer_objects()
Normal_data_numeric = Normal_data_numeric.infer_objects()
Cancer_data_numeric = Cancer_data_numeric.apply(pd.to_numeric, errors='coerce')
Normal_data_numeric = Normal_data_numeric.apply(pd.to_numeric, errors='coerce')
Cancer_data_numeric.fillna(0, inplace=True)
Normal_data_numeric.fillna(0, inplace=True)

# Set the index to the Sample column
Cancer_data_numeric.set_index(Cancer_data['Sample'], inplace=True)
Normal_data_numeric.set_index(Nornal_data['Sample'], inplace=True)

# print(Cancer_data_numeric.describe())
# Add a column to label data as 'Cancer' or 'Normal'
Cancer_data_numeric['Type'] = 'Cancer'
Normal_data_numeric['Type'] = 'Normal'

# Combine both datasets for plotting
combined_data = pd.concat([Cancer_data_numeric, Normal_data_numeric])

# Convert wide-format to long-format for plotting with hue
melted_data = pd.melt(combined_data.reset_index(), id_vars=['Sample', 'Type'], 
                      var_name='Genomic Region', value_name='Value')

# Plot using hue to differentiate between cancer and normal samples
# plt.figure(1, figsize=(15, 9))

# sns.stripplot(data=melted_data[melted_data['Type'] == 'Cancer'], x='Value', y='Genomic Region', 
#               jitter=True, size=5, marker="o", alpha=0.5, color="blue")

# sns.stripplot(data=melted_data[melted_data['Type'] == 'Normal'], x='Value', y='Genomic Region', 
#               jitter=True, size=5, marker="o", alpha=0.8, color="red")

# blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Cancer', alpha=0.5)
# red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Normal', alpha=0.8)

# plt.legend(handles=[blue_dot, red_dot], title='Sample Type', loc='upper right')

# plt.yticks([])
# plt.title('Genomic Regions Data Distribution', fontsize=18)
# plt.xlabel('Genomic Regions Values', fontsize=18)
# plt.ylabel('Genomic Regions', fontsize=18)
# plt.show()

## Plot for Stage IV and Stage I

stage_iv_numeric = stage_iv.drop(columns=non_numeric_cols, errors='ignore')
stage_i_numeric = stage_i.drop(columns=non_numeric_cols, errors='ignore')


stage_iv_numeric = stage_iv_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
stage_i_numeric = stage_i_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)


plt.figure(2, figsize=(15, 9))
# sns.boxplot(data=stage_iv_numeric, orient='h', palette="coolwarm")
plt.subplot(2, 1, 1)
plt.xlim(-40, 110)
sns.stripplot(data=stage_iv_numeric, orient='h', color="blue", jitter=True, size=5, marker="o", alpha=0.5)
plt.yticks([])
plt.title('Boxplot of Genomic Regions for Stage IV', fontsize=18)
# plt.xlabel('Genomic Regions Values', fontsize=18)
plt.ylabel('Genomic Regions', fontsize=18)
# plt.show()
plt.subplot(2, 1, 2)
# plt.figure(3, figsize=(15, 9))
# sns.boxplot(data=stage_i_numeric, orient='h', palette="coolwarm")
plt.xlim(-40, 110)
sns.stripplot(data=stage_i_numeric, orient='h', color="blue", jitter=True, size=5, marker="o", alpha=0.5)
plt.yticks([])
plt.title('Boxplot of Genomic Regions for Stage I', fontsize=18)
plt.xlabel('Genomic Regions Values', fontsize=18)
plt.ylabel('Genomic Regions', fontsize=18)
plt.show()