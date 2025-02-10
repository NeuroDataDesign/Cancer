import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Ensure figures directory exists
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)

def test_model(model, model_name, X_test, y_test, timestamp):
    """ Test a trained model and plot the ROC curve with S@98 annotation. """
    print(f"Testing {model_name} model...")

    # **Use direct prediction**
    print("Using direct prediction...")
    posterior_mean = model.predict_proba(X_test)  # (n_samples, 2)

    # **Ensure posterior_mean is 1D**
    posterior_final = posterior_mean[:, 1] if posterior_mean.ndim == 2 else posterior_mean.ravel()

    # **Handle NaN values**
    if np.isnan(posterior_final).any():
        print("Detected NaN values, replacing with median.")
        posterior_final = np.nan_to_num(posterior_final, nan=np.nanmedian(posterior_final))

    # **Compute ROC and AUC**
    fpr, tpr, _ = roc_curve(y_test, posterior_final, pos_label=1)
    auc_score = roc_auc_score(y_test, posterior_final)

    # **Compute S@98 (maximum TPR where FPR ≤ 0.02)**
    S98 = np.max(tpr[fpr <= 0.02]) if np.any(fpr <= 0.02) else 0

    # **Find the corresponding FPR for S@98**
    fpr_S98 = fpr[fpr <= 0.02][-1] if np.any(fpr <= 0.02) else None

    # **Plot ROC curve**
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    # **Mark S@98 point**
    if fpr_S98 is not None:
        plt.scatter(fpr_S98, S98, color='red', label=f'S@98: {S98:.3f}', zorder=3)
        plt.annotate(f'S@98: {S98:.3f}', 
                     (fpr_S98, S98), 
                     textcoords="offset points", 
                     xytext=(-15,10), 
                     ha='center', fontsize=18, color='red')

    # **Plot settings**
    plt.xlabel('False Positive Rate (FPR)', fontsize=20)
    plt.ylabel('True Positive Rate (TPR)', fontsize=20)
    plt.title(f'{model_name} ROC Curve', fontsize=24)
    plt.legend()
    plt.grid(alpha=0.5)

    # **Save figure with timestamp**
    save_path = f"{fig_dir}/{model_name}_ROC_{timestamp}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"ROC curve saved to {save_path}")

    return S98