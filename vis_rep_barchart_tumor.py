import matplotlib.pyplot as plt
import numpy as np

tumor_types = ["Total", "Lung", "Liver", "Pancreas", "Ovary", "Stomach", "Kidney", "Colorectal", "Breast", "Esophagus"]
methods = ["knn", "lr", "svm", "rf", "might"]

# data
mean_values = [
    [0.241, 0.56, 0.31, 0.24, 0.26, 0.085, 0.09, 0.18, 0.13, 0.2],  # knn
    [0.339, 0.48, 0.51, 0.3, 0.48, 0.2, 0.16, 0.23, 0.17, 0.36],   # lr
    [0.465, 0.72, 0.67, 0.5, 0.6, 0.32, 0.2, 0.24, 0.24, 0.72],    # svm
    [0.398, 0.73, 0.6, 0.42, 0.47, 0.29, 0.22, 0.19, 0.21, 0.69],  # rf
    [0.309, 0.66, 0.4, 0.33, 0.36, 0.26, 0.15, 0.1, 0.16, 0.38]    # might
]

variance_values = [
    [0.001, 0.0022, 0.0036, 0.00075, 0.0015, 0.00028, 0.0005, 0.0007, 0.00038, 0.0079],  # knn
    [0.0007, 0.003, 0.0018, 0.0011, 0.00058, 0.0015, 0.001, 0.00074, 0.00082, 0.021],   # lr
    [0.0009, 0.00039, 0.0021, 0.0012, 0.00089, 0.0025, 0.0013, 0.00039, 0.00086, 0.0074], # svm
    [0.0007, 0.00022, 0.0018, 0.00061, 0.00095, 0.0013, 0.0022, 0.00028, 0.00059, 0.01], # rf
    [3e-05, 0.00011, 0.00085, 0.00015, 0.00012, 0.0012, 0.00019, 1.7e-05, 6.6e-05, 0.0029] # might
]

# (deviation)
deviation_values = [[np.sqrt(v) for v in row] for row in variance_values]

# set up plot
x = np.arange(len(tumor_types))  # the label locations
width = 0.15  

fig, ax = plt.subplots(figsize=(12, 8))

# plot bars
colors = ['b', 'g', 'r', 'orange', 'purple']  # colors for the bars
for i, (method, mean, deviation) in enumerate(zip(methods, mean_values, deviation_values)):
    ax.bar(
        x + i * width - (width * len(methods) / 2),  # shift the bars
        mean,
        width,
        yerr=deviation,  # add deviation bars
        label=method,
        color=colors[i],
        alpha=0.8,
        capsize=5
    )

# set labels
# ax.set_xlabel("Tumor Types", fontsize=24)
ax.set_ylabel("Mean ± Deviation", fontsize=24)
ax.set_title("Model Performance(S@98) Across Tumor Types", fontsize=26)
ax.set_xticks(x)
ax.set_xticklabels(tumor_types, rotation=60, ha="right", fontsize=20)
ax.legend(title="Methods", fontsize=20, title_fontsize=20, loc="upper left")

# show grid
ax.grid(alpha=0.3)

# save the plot
ouput_dir = "./Outcome"
output_file = f"{ouput_dir}/model_performance_across_tumor_types.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Bar chart saved at: {output_file}")