from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 获取样本列表
sample_list_file = "AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]

# 获取癌症样本的 sample_ids
Cancer_samples = sample_list[sample_list["cohort"] != "PanelOfNormals"]["sample_id"]
print(len(Cancer_samples))

# 读取数据
data = pd.read_csv("WiseCondorX.Wise-1.csv", low_memory=False)

# 打印一些数据的信息以便理解
# print(data.columns)

# 提取癌症样本的数据
Cancer_data = data[data['Sample'].isin(Cancer_samples)]

# 删除非数值型列
non_numeric_cols = ['Run', 'Sample','Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
Cancer_data_numeric = Cancer_data.drop(columns=non_numeric_cols, errors='ignore')

# 删除所有全为 NaN 的列
nan_cols = Cancer_data_numeric.isnull().all(axis=0).to_numpy()
Cancer_data_numeric = Cancer_data_numeric.loc[:, ~nan_cols]

# 确保所有数据为数值类型
Cancer_data_numeric = Cancer_data_numeric.infer_objects()

# 检查并确保所有值都是数值型
Cancer_data_numeric = Cancer_data_numeric.apply(pd.to_numeric, errors='coerce')

# 将 NaN 值填充为 0（若有任何非数值转换后的缺失值）
Cancer_data_numeric.fillna(0, inplace=True)

# 设置样本 ID 作为行索引
Cancer_data_numeric.set_index(Cancer_data['Sample'], inplace=True)

print(Cancer_data_numeric.describe())

# 使用箱线图来显示各列的分布及极端值
plt.figure(figsize=(20, 8))
sns.boxplot(data=Cancer_data_numeric, orient='h', palette="Set3")
plt.title('Boxplot of Genomic Regions Data Distribution')
plt.xlabel('Genomic Regions Values')
plt.ylabel('Genomic Regions')
plt.show()