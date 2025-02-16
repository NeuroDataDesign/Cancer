import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Ensure figures directory exists
# fig_dir = "./figures"
# os.makedirs(fig_dir, exist_ok=True)

def test_model(model, model_name, X_test, y_test, timestamp):
    """ Test a trained model and plot the ROC curve with S@98 annotation. """
    print(f"Testing {model_name} model...")

    # **Ensure model uses all CPU cores if possible**
    if hasattr(model, "predict_proba"):
        posterior_mean = model.predict_proba(X_test)

        # **Handle case where only one probability column is returned**
        if posterior_mean.shape[1] == 1:
            print(f"Warning: {model_name} only returns one probability column. Using column 0 instead.")
            posterior_final = posterior_mean[:, 0]
        else:
            posterior_final = posterior_mean[:, 1]
    else:
        print(f"Warning: {model_name} does not support `predict_proba()`. Using `predict()` instead.")
        posterior_final = model.predict(X_test)  # 直接使用类别预测
    
    # **Handle NaN values efficiently**
    if np.isnan(posterior_final).any():
        print("Warning: NaN values detected in predictions! Replacing with median.")
        posterior_final = np.where(np.isnan(posterior_final), np.nanmedian(posterior_final), posterior_final)

    # **Check if y_test contains both 0 and 1**
    unique_labels = np.unique(y_test)
    if len(unique_labels) < 2:
        print(f"Warning: y_test only contains one class {unique_labels}, ROC curve cannot be computed.")
        return None
    
    # **Compute ROC and AUC**
    try:
        fpr, tpr, _ = roc_curve(y_test, posterior_final, pos_label=1)
        auc_score = roc_auc_score(y_test, posterior_final)
    except ValueError as e:
        print(f"Error computing ROC: {e}")
        return None
    # **Compute S@98 (maximum TPR where FPR ≤ 0.02)**
    idx = np.searchsorted(fpr, 0.02, side="right")  # Efficiently find last index ≤ 0.02
    S98 = tpr[idx-1] if idx > 0 else 0
    # S98 = np.max(tpr[fpr <= 0.02]) if np.any(fpr <= 0.02) else 0
    # fpr_S98 = fpr[fpr <= 0.02][-1] if np.any(fpr <= 0.02) else None
    
    # Print S@98
    # print(f"Model: {model_name}", f"AUC Score: {auc_score:.4f}", f"S@98: {S98:.4f}")

    # **Plot ROC curve**
    # plt.figure(figsize=(8, 8))  # Reduce figure size for efficiency
    # plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    # plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # **Only plot S@98 if it exists**
    # if fpr_S98 is not None:
    #     plt.scatter(fpr_S98, S98, color='red', label=f'S@98: {S98:.3f}', zorder=3)
    #     plt.annotate(f'S@98: {S98:.3f}', 
    #                  (fpr_S98, S98), 
    #                  textcoords="offset points", 
    #                  xytext=(-15,10), 
    #                  ha='center', fontsize=14, color='red')

    # **Plot settings**
    # plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    # plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    # plt.title(f'{model_name} ROC Curve', fontsize=18)
    # plt.legend()
    # plt.grid(alpha=0.5)

    # **Efficiently save figure**
    # save_path = f"{fig_dir}/{model_name}_ROC_{timestamp}.png"
    # plt.draw()  # Precompute the figure
    # plt.savefig(save_path, dpi=150)  # Reduce dpi for speed
    # plt.close()

    # print(f"ROC curve saved to {save_path}")

    return S98