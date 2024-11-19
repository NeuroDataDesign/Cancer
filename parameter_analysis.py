import pandas as pd
import matplotlib.pyplot as plt

file_path = "Cancer_HF.csv"  #
data = pd.read_csv(file_path)

grouped = data.groupby("Honest_Fraction").agg(['mean', 'std'])

methods = ["SVM", "MIGHT", "SPO-MIGHT"]
colors = ["blue", "red", "green"]
markers = ["o", "^", "s"]

plt.figure(figsize=(10, 6))


for method, color, marker in zip(methods, colors, markers):
    means = grouped[(method, "mean")]
    stds = grouped[(method, "std")]
    plt.errorbar(grouped.index, means, yerr=stds, label=method, color=color,
                 marker=marker, capsize=5, linestyle='-', linewidth=1)


# plt.title("Performance of SVM, MIGHT, SPO-MIGHT", fontsize=14)
# plt.xlabel("MAX Features", fontsize=12)
# plt.ylabel("s@98", fontsize=12)
plt.xticks(grouped.index, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.tight_layout()
plt.show()