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

random_state = 57

# **1️⃣ 读取数据**
data = pd.read_csv("./data/WiseCondorX.Wise-1.csv")
data['Cancer Status'] = data['Cancer Status'].replace({"Healthy": 0, "Cancer": 1}).astype(int)
data = data.dropna(axis=1, how='all').fillna(0)

# **2️⃣ 统一 `Stage` 映射**
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

# **3️⃣ 去除非特征列**
non_feature_cols = ['Run', 'Library', 'Library volume (uL)',
                    'Library Volume', 'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
data = data.drop(columns=[col for col in non_feature_cols if col in data.columns])

# **4️⃣ 构建测试集**
cancer_types_test = {'Stomach', 'Pancreas', 'Esophagus', 'Lung', 'Liver', 'Ovary'}
test_set = data[(data['Stage'] == 'I') & (data['Tumor type'].isin(cancer_types_test))]
normal_samples = data[data['Stage'] == 'Normal']
normal_test_samples = normal_samples.sample(n=100, random_state=random_state)
test_set = pd.concat([test_set, normal_test_samples])

# **5️⃣ 平衡训练集**
train_set = data.drop(test_set.index)
train_counts = train_set["Tumor type"].value_counts()
min_samples_per_cancer = max(2, min(train_counts))  # 选最少的癌症样本数作为基准

# **防止某些癌症数据过少或过多**
balanced_train_set = []
for cancer_type, count in train_counts.items():
    cancer_samples = train_set[train_set["Tumor type"] == cancer_type]

    # **下采样：如果样本太多，最多取 5 倍于最少类别**
    if count > min_samples_per_cancer * 5:
        cancer_samples = cancer_samples.sample(n=min_samples_per_cancer * 5, random_state=random_state)

    # **上采样：如果样本过少，增加到 `min_samples_per_cancer`**
    elif count < min_samples_per_cancer:
        cancer_samples = cancer_samples.sample(n=min_samples_per_cancer, random_state=random_state, replace=True)

    balanced_train_set.append(cancer_samples)

# **合并所有类别，并重新打乱**
train_set = pd.concat(balanced_train_set).sample(frac=1, random_state=random_state)

# **6️⃣ 训练 & 测试**
def train_and_test(model_name, cancer_type):
    """ 训练并测试模型，返回 S@98 """
    print(f"\nTraining {model_name} for {cancer_type}...")

    # **筛选当前癌症类型的训练数据**
    cancer_samples = train_set[(train_set['Tumor type'] == cancer_type) & (train_set['Cancer Status'] == 1)]
    healthy_samples = train_set[train_set['Cancer Status'] == 0]

    # **保证正负样本均衡**
    num_cancer = min(1000, len(cancer_samples))
    num_healthy = min(len(healthy_samples), num_cancer)  # Healthy 不超过 Cancer 的数量

    cancer_samples = cancer_samples.sample(n=num_cancer, random_state=random_state, replace=True if num_cancer > len(cancer_samples) else False)
    healthy_samples = healthy_samples.sample(n=num_healthy, random_state=random_state, replace=True if num_healthy > len(healthy_samples) else False)

    train_subset = pd.concat([cancer_samples, healthy_samples]).sample(frac=1, random_state=random_state)

    X_train = train_subset.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type'])
    y_train = train_subset['Cancer Status'].values.ravel()

    X_test = test_set.drop(columns=['Cancer Status', 'Stage', 'Sample', 'Tumor type'])
    y_test = test_set['Cancer Status'].values.ravel()

    model = train_model(model_name, X_train, y_train, timestamp=None)
    S98 = test_model(model, model_name, X_test, y_test, timestamp=None)

    return model_name, cancer_type, S98

# **7️⃣ 并行训练**
MODEL_NAMES = ["MIGHT", "SPO-MIGHT", "SPORF"]

results = Parallel(n_jobs=40)(
    delayed(train_and_test)(model_name, cancer_type)
    for model_name in MODEL_NAMES
    for cancer_type in cancer_types_test
)

gc.collect()
get_reusable_executor().shutdown(wait=True)

# **8️⃣ 输出最终结果**
print("\nFinal Results in Tumor Type:")
for model_name, cancer_type, S98 in results:
    print(f"Model: {model_name}, Cancer Type: {cancer_type}, S@98: {S98:.4f}")