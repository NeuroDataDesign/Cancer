import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from treeple.tree import MultiViewDecisionTreeClassifier, ObliqueDecisionTreeClassifier
from treeple import HonestForestClassifier
from treeple import ObliqueRandomForestClassifier, PatchObliqueRandomForestClassifier
from treeple.stats import PermutationHonestForestClassifier, build_oob_forest
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

n_estimators = 1000
max_features = 0.3

MODEL_NAMES = {
    "might": {
        "n_estimators": n_estimators,
        "honest_fraction": 0.5,
        "n_jobs": 40,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "max_features": 0.3,
        "tree_estimator": MultiViewDecisionTreeClassifier(),
    },
    "rf": {
        "n_estimators": int(n_estimators / 5),
        "max_features": 0.3,
    },
    "knn": {
        # XXX: above, we use sqrt of the total number of samples to allow
        # scaling wrt the number of samples
        # "n_neighbors": 5,
    },
    "svm": {
        "probability": True,
    },
    "lr": {
        "max_iter": 1000,
        "penalty": "l1",
        "solver": "liblinear",
    },
    "HFODT": {
        "n_estimators": n_estimators,
        "honest_fraction": 0.5,
        "n_jobs": 40,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "max_features": 0.3,
        "tree_estimator": ObliqueDecisionTreeClassifier(),
    },
}
might_kwargs = MODEL_NAMES["might"]
HFODT_kwargs = MODEL_NAMES["HFODT"]

def generate_data(n_samples=1000, n_features=10, random_seed=42):
    X_real, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=4,  # 所有真实特征对目标变量有贡献
        n_redundant=0,
        random_state=random_seed
    )

    # Step 2: 生成噪声特征
    X_noise = np.random.normal(0, 1, size=(n_samples, (n_features - 4)))  # 生成高斯噪声特征

    # Step 3: 将真实特征和噪声特征组合
    X_combined = np.hstack((X_real, X_noise))

    return X_combined, y

def stratified_train_ml(clf, X, y):
    n_samples = X.shape[0]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    POS = np.zeros((len(y), 3))

    for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        ### Split Training Set into Fitting Set (40%) and Calibarating Set (40%)
        train_idx = np.arange(
            X_train.shape[0]
        )  # use index array to split, so we can use the same index for the permuted array as well
        fit_idx, cal_idx = train_test_split(
            train_idx, test_size=0.5, random_state=idx, stratify=y_train
        )
        X_fit, X_cal, y_fit, y_cal = (
            X_train[fit_idx],
            X_train[cal_idx],
            y_train[fit_idx],
            y_train[cal_idx],
        )

        POS[test_ix, 0] = y_test
        clf.fit(X_fit, y_fit)
        if X_cal.shape[0] <= 1000:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="sigmoid"
            )
        else:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="isotonic"
            )
        calibrated_model.fit(X_cal, y_cal)
        posterior = calibrated_model.predict_proba(X_test)

        POS[test_ix, 1:] = posterior
    return clf, POS

# Define feature sizes to test
feature_sizes = [4] + [2 ** i for i in range(3, 13)]  # from 4 to 2048 (4, 8, 16, ..., 2048)
accuracies_knn = []
accuracies_rf = []
accuracies_svm = []
accuracies_lr = []
accuracies_might = []
accuracies_SPORF = []
accuracies_SPmight = []




# Loop over each feature size
for n_features in feature_sizes:
    X, y = generate_data(n_samples=1000, n_features=n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for model_name in ['might', 'rf', 'knn', 'lr', 'svm', 'SPORF', 'SPmight']:
        if model_name == 'might':
            est = HonestForestClassifier(**might_kwargs)

        elif model_name == "rf":
            est = RandomForestClassifier(**MODEL_NAMES[model_name], n_jobs=40)

        elif "knn" in model_name:
            est = KNeighborsClassifier(n_neighbors=int(np.sqrt(X.shape[0]) + 1), )

        elif model_name == "svm":
            est = SVC(**MODEL_NAMES[model_name])

        elif model_name == "lr":
            est = LogisticRegression(**MODEL_NAMES[model_name])

        elif model_name == "SPORF":
            est = ObliqueRandomForestClassifier(n_estimators=6000, feature_combinations=2.3)

        elif model_name == "SPORF_might":
            est = HonestForestClassifier(**HFODT_kwargs)

        if model_name in ['might', 'SPORF_might']:
            est, posterior_arr = build_oob_forest(est, X, y, verbose=False, )
            POS = np.nanmean(posterior_arr, axis=0)
        else:
            est, posterior_arr = stratified_train_ml(est, X, y)
            POS = posterior_arr

        fpr, tpr, thresholds = roc_curve(y, POS[:, -1], pos_label=1, drop_intermediate=False, )

        S98 = np.max(tpr[fpr <= 0.02])


        if model_name == 'might':
            accuracies_might.append(S98)

        elif model_name == "rf":
            accuracies_rf.append(S98)

        elif model_name == "knn":
            accuracies_knn.append(S98)

        elif model_name == "svm":
            accuracies_svm.append(S98)

        elif model_name == "lr":
            accuracies_lr.append(S98)

        elif model_name == "SPORF":
            accuracies_SPORF.append(S98)

        elif model_name == "SPmight":
            accuracies_SPmight.append(S98)
    print(n_features)
    # # Train KNN
    # knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(X.shape[0]) + 1), )
    # est_knn, posterior_arr = stratified_train_ml(knn, X, y)
    # POS = posterior_arr
    # fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, POS[:, -1], pos_label=1, drop_intermediate=False, )
    # S98_knn = np.max(tpr_knn[fpr_knn <= 0.02])
    # accuracies_knn.append(S98_knn)
    # # knn.fit(X_train, y_train)
    # # y_pred_knn = knn.predict(X_test)
    # # acc_knn = accuracy_score(y_test, y_pred_knn)
    # # accuracies_knn.append(acc_knn)
    #
    #
    # # Train Random Forest
    # rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    # est_rf, posterior_arr = stratified_train_ml(rf, X, y)
    # POS = posterior_arr
    # fpr_knn, tpr_knn, thresholds_knn = roc_curve(y, POS[:, -1], pos_label=1, drop_intermediate=False, )
    # S98_knn = np.max(tpr_knn[fpr_knn <= 0.02])
    # accuracies_knn.append(S98_knn)
    # # rf.fit(X_train, y_train)
    # # y_pred_rf = rf.predict(X_test)
    # # acc_rf = accuracy_score(y_test, y_pred_rf)
    # # accuracies_rf.append(acc_rf)
    #
    # # Train SVM
    # svm = SVC(random_state=42)
    # svm.fit(X_train, y_train)
    # y_pred_svm = svm.predict(X_test)
    # acc_svm = accuracy_score(y_test, y_pred_svm)
    # accuracies_svm.append(acc_svm)
    #
    # lr = LogisticRegression(max_iter=1000,
    #     penalty="l1",
    #     solver="liblinear")
    # lr.fit(X_train, y_train)
    # y_pred_lr = lr.predict(X_test)
    # acc_lr = accuracy_score(y_test, y_pred_lr)
    # accuracies_lr.append(acc_lr)
    #
    # might = HonestForestClassifier(**might_kwargs)
    # might.fit(X_train, y_train)
    # y_pred_might = might.predict(X_test)
    # acc_might = accuracy_score(y_test, y_pred_might)
    # accuracies_might.append(acc_might)
    #
    # SPORF = ObliqueRandomForestClassifier(n_estimators=1000, feature_combinations=2.3)
    # SPORF.fit(X_train, y_train)
    # y_pred_SPORF = SPORF.predict(X_test)
    # acc_SPORF = accuracy_score(y_test, y_pred_SPORF)
    # accuracies_SPORF.append(acc_SPORF)
    #
    # SPmight = HonestForestClassifier(**HFODT_kwargs)
    # SPmight.fit(X_train, y_train)
    # y_pred_SPmight = SPmight.predict(X_test)
    # acc_SPmight = accuracy_score(y_test, y_pred_SPmight)
    # accuracies_SPmight.append(acc_SPmight)
    #


# Plotting the accuracy curves
plt.figure(figsize=(12, 8))
plt.plot(feature_sizes, accuracies_knn, marker='o', label='KNN')
plt.plot(feature_sizes, accuracies_rf, marker='o', label='Random Forest')
plt.plot(feature_sizes, accuracies_svm, marker='o', label='SVM')
plt.plot(feature_sizes, accuracies_lr, marker='o', label='LR')
plt.plot(feature_sizes, accuracies_might, marker='o', label='might')
plt.plot(feature_sizes, accuracies_SPORF, marker='o', label='SPORF')
plt.plot(feature_sizes, accuracies_SPmight, marker='o', label='SPmight')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.xticks([4, 2048, 4096])  # 设置指定刻度
plt.xlim(4, 4096)
# plt.xscale('log')  # Log scale for feature sizes
plt.title('Model Accuracy vs Number of Features')
plt.legend()
plt.grid(True)
plt.show()
