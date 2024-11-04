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
Normal_samples = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]
print(len(Cancer_samples))
print(len(Normal_samples))

# Read the data
data = pd.read_csv("WiseCondorX.Wise-1.csv", low_memory=False)

# unique_tumor = data['Tumor type'].unique()
unique_tumor = data['Tumor type'].unique()
print("Tumor type:", unique_tumor)

for tumor in unique_tumor:
    print(tumor, data[data['Tumor type'] == tumor].shape)
    
# Tumor type: ['Stomach' 'Colorectal' 'Pancreas' 'Liver' 'Esophagus' 'Lung' 'Ovary'
#  'Breast' 'Healthy' 'Normal']
# stage_counts = data['Stage'].value_counts(dropna=False) 
# print("Cancer stage counts:\n", stage_counts)

# Filter the data to include only the cancer samples
# stage_iv = data[data['Stage'].isin(['IV'])]
# stage_i = data[data['Stage'].isin(['I'])]

# print(data.columns)

# Filter the data to include only the cancer samples
# Cancer_data = data[data['Sample'].isin(Cancer_samples)]
# Nornal_data = data[data['Sample'].isin(Normal_samples)]

# Drop columns
non_numeric_cols = ['Run', 'Sample','Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
genomic_regions = [
    '17:43000001-44000000', '1:121000001-122000000', '17:20000001-21000000', 
    '8:140000001-141000000', '20:60000001-61000000', '20:59000001-60000000', 
    '5:2000001-3000000', '10:42000001-43000000', '4:68000001-69000000', 
    '20:58000001-59000000', '1:25000001-26000000', '8:143000001-144000000', 
    '2:131000001-132000000', '2:98000001-99000000', '8:142000001-143000000', 
    '3:93000001-94000000', '5:6000001-7000000', '1:22000001-23000000', 
    '8:63000001-64000000', '8:136000001-137000000'
]

# Filter the data to include only the cancer samples
tumor_types = {
    'Stomach': data[data['Tumor type'] == 'Stomach'][genomic_regions],
    'Colorectal': data[data['Tumor type'] == 'Colorectal'][genomic_regions],
    'Pancreas': data[data['Tumor type'] == 'Pancreas'][genomic_regions],
    'Liver': data[data['Tumor type'] == 'Liver'][genomic_regions],
    'Esophagus': data[data['Tumor type'] == 'Esophagus'][genomic_regions],
    'Lung': data[data['Tumor type'] == 'Lung'][genomic_regions],
    'Ovary': data[data['Tumor type'] == 'Ovary'][genomic_regions],
    'Breast': data[data['Tumor type'] == 'Breast'][genomic_regions],
    'Kidney': data[data['Tumor type'] == 'Kidney'][genomic_regions]
}

for tumor, df in tumor_types.items():
    tumor_types[tumor] = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# 为每种肿瘤类型单独绘制 boxplot 并叠加 stripplot
for tumor, df in tumor_types.items():
    plt.figure(figsize=(15, 10))
    
    # 绘制 boxplot
    sns.boxplot(data=df, orient='h', color="lightblue", fliersize=2, linewidth=1.5,
                boxprops=dict(facecolor="lightblue", edgecolor="blue", linewidth=2),
                whiskerprops=dict(color="blue", linewidth=2),
                capprops=dict(color="blue", linewidth=2))
    
    # 在 boxplot 上叠加 stripplot，使用透明度展示数据点的密度
    sns.stripplot(data=df, orient='h', color="darkblue", jitter=True, size=2, alpha=0.4)
    
    plt.title(f'Genomic Regions Data Distribution for {tumor}', fontsize=16)
    plt.xlabel('Genomic Regions Values', fontsize=14)
    # plt.yticks([])  
    plt.show()


## Plot for Stage IV and Stage I

# stage_iv_numeric = stage_iv.drop(columns=non_numeric_cols, errors='ignore')
# stage_i_numeric = stage_i.drop(columns=non_numeric_cols, errors='ignore')


# stage_iv_numeric = stage_iv_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)
# stage_i_numeric = stage_i_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)

# stage_iv_numeric['Type'] = 'Stage IV'
# stage_i_numeric['Type'] = 'Stage I'

# Combine Stage IV and Stage I data for plotting
# combined_data_stage = pd.concat([stage_iv_numeric, stage_i_numeric])
# print(combined_data_stage.describe())

# Convert wide-format to long-format for plotting
# melted_data_stage = pd.melt(combined_data_stage.reset_index(), id_vars=['Type'], 
#                             var_name='Genomic Region', value_name='Value')



# Plot using hue to differentiate between Stage IV and Stage I
# plt.figure(figsize=(15, 9))

# # Plot Stage IV data (in red) and Stage I data (in blue) on the same figure
# sns.stripplot(data=melted_data_stage, x='Value', y='Genomic Region', hue='Type', 
#               jitter=True, size=5, marker="o", alpha=0.7, palette={"Stage IV": "red", "Stage I": "blue"})

# # plt.yticks([])  # Hide y-axis labels
# plt.title('Genomic Regions Data Distribution for Stage I and IV', fontsize=32)
# plt.xlabel('Genomic Regions Values', fontsize=32)
# plt.ylabel('Genomic Regions', fontsize=32)
# plt.legend(title='Stage Type', loc='upper right')
# plt.show()


# Cancer_data_numeric = Cancer_data.drop(columns=non_numeric_cols, errors='ignore')
# Normal_data_numeric = Nornal_data.drop(columns=non_numeric_cols, errors='ignore')
# nan_cols = Cancer_data_numeric.isnull().all(axis=0).to_numpy()
# nan_cols_normal = Normal_data_numeric.isnull().all(axis=0).to_numpy()
# Cancer_data_numeric = Cancer_data_numeric.loc[:, ~nan_cols]
# Normal_data_numeric = Normal_data_numeric.loc[:, ~nan_cols_normal]
# Cancer_data_numeric = Cancer_data_numeric.infer_objects()
# Normal_data_numeric = Normal_data_numeric.infer_objects()
# Cancer_data_numeric = Cancer_data_numeric.apply(pd.to_numeric, errors='coerce')
# Normal_data_numeric = Normal_data_numeric.apply(pd.to_numeric, errors='coerce')
# Cancer_data_numeric.fillna(0, inplace=True)
# Normal_data_numeric.fillna(0, inplace=True)

# Set the index to the Sample column
# Cancer_data_numeric.set_index(Cancer_data['Sample'], inplace=True)
# Normal_data_numeric.set_index(Nornal_data['Sample'], inplace=True)

# print(Cancer_data_numeric.describe())
# Add a column to label data as 'Cancer' or 'Normal'
# Cancer_data_numeric['Type'] = 'Cancer'
# Normal_data_numeric['Type'] = 'Normal'

# Combine both datasets for plotting
# combined_data = pd.concat([Cancer_data_numeric, Normal_data_numeric])

# Convert wide-format to long-format for plotting with hue
# melted_data = pd.melt(combined_data.reset_index(), id_vars=['Sample', 'Type'], 
#                       var_name='Genomic Region', value_name='Value')

# Plot using hue to differentiate between cancer and normal samples
# plt.figure(1)

# sns.stripplot(data=melted_data[melted_data['Type'] == 'Cancer'], x='Value', y='Genomic Region', 
#               jitter=True, size=5, marker="o", alpha=0.5, color="blue")

# sns.stripplot(data=melted_data[melted_data['Type'] == 'Normal'], x='Value', y='Genomic Region', 
#               jitter=True, size=5, marker="o", alpha=0.8, color="red")

# blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='Cancer', alpha=0.5)
# red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Normal', alpha=0.1)

# plt.legend(handles=[blue_dot, red_dot], title='Sample Type', loc='upper right')

# plt.yticks([])
# plt.title('Genomic Regions Data Distribution', fontsize=32)
# plt.xlabel('Genomic Regions Values', fontsize=32)
# plt.ylabel('Genomic Regions', fontsize=32)
# plt.show()
