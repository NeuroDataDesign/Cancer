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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('future.no_silent_downcasting', True)

random_state = 507

sample_list_file = "data/AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]

# get the sample_ids where cohort is Cohort1
cohort1 = sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"]
cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]

# Load Data and Process Cancer Status
def get_X_y(f, root, cohort=[], verbose=False):
    df = pd.read_csv(root + f)
    non_features = [
        "Run",
        "Sample",
        "Library",
        "Cancer Status",
        "Tumor type",
        "Stage",
        "Library volume (uL)",
        "Library Volume",
        "UIDs Used",
        "Experiment",
        "P7",
        "P7 Primer",
        "MAF",
    ]
    sample_ids = df["Sample"]
    # if sample is contains "Run" column, remove it
    for i, sample_id in enumerate(sample_ids):
        if "." in sample_id:
            sample_ids[i] = sample_id.split(".")[1]
    target = "Cancer Status"
    y = df[target]
    # convert the labels to 0 and 1
    y = y.replace("Healthy", 0)
    y = y.replace("Cancer", 1)
    # remove the non-feature columns if they exist
    for col in non_features:
        if col in df.columns:
            df = df.drop(col, axis=1)
    nan_cols = df.isnull().all(axis=0).to_numpy()
    # drop the columns with all nan values
    df = df.loc[:, ~nan_cols]
    # if cohort is not None, filter the samples
    if cohort is not None:
        # filter the rows with cohort1 samples
        X = df[sample_ids.isin(cohort)]
        y = y[sample_ids.isin(cohort)]
    else:
        X = df
    if "Wise" in f:
        # replace nans with zero
        X = X.fillna(0)
    # impute the nan values with the mean of the column
    X = X.fillna(X.mean(axis=0))
    # check if there are nan values
    # nan_rows = X.isnull().any(axis=1)
    nan_cols = X.isnull().all(axis=0)
    # remove the columns with all nan values
    X = X.loc[:, ~nan_cols]
    if verbose:
        if nan_cols.sum() > 0:
            print(f)
            print(f"nan_cols: {nan_cols.sum()}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        else:
            print(f)
            print(f"X shape: {X.shape}, y shape: {y.shape}")
    # X = X.dropna()
    # y = y.drop(nan_rows.index)

    return X, y

X, y = get_X_y("WiseCondorX.Wise-1.csv", root="data/", cohort=cohort2, verbose=True)

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# pca = PCA(n_components=300)
# X_pca = pca.fit_transform(X_scaled)

n_splits = 5

# 5-Fold Cross Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def train_and_test_fold(model_name, fold, train_idx, test_idx):
    """ Train and test a model on a single fold """
    print(f"\nTraining {model_name} - Fold {fold + 1}/{n_splits}...")

    # X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].astype(int).values.ravel(), y.iloc[test_idx].astype(int).values.ravel()

    model = train_model(model_name, X_train, y_train, timestamp=None)
    S98 = test_model(model, model_name, X_test, y_test, timestamp=None)

    return model_name, fold, S98 

# Models to train and test
MODEL_NAMES = ["MIGHT", "SPO-MIGHT", "SPORF"]

# Train and test models in parallel
results = Parallel(n_jobs=40)(
    delayed(train_and_test_fold)(model_name, fold, train_idx, test_idx)
    for model_name in MODEL_NAMES
    # for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca))
    for fold, (train_idx, test_idx) in enumerate(kf.split(X))
)

gc.collect()

# Store S@98 scores for each model
s98_scores = {model: [] for model in MODEL_NAMES}

for model_name, fold, S98 in results:
    s98_scores[model_name].append((fold, S98))

# Print final results
print("\nFinal 5-Fold Cross Validation Results:")
for model_name in MODEL_NAMES:
    print(f"\nModel: {model_name}")
    
    for fold, S98 in sorted(s98_scores[model_name]):
        print(f"  Fold {fold + 1}: S@98 = {S98:.4f}") 

    # Compute mean and standard deviation of S@98 scores
    mean_s98 = np.mean([S98 for _, S98 in s98_scores[model_name]])
    std_s98 = np.std([S98 for _, S98 in s98_scores[model_name]])

    print(f"  Mean S@98: {mean_s98:.4f} ± {std_s98:.4f}")