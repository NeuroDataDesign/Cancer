import matplotlib.pyplot as plt
import numpy as np
import os

# Data
cohorts = ["Cohort 1", "Cohort 2"]
models = ["svm", "rf", "SPORF", "MIGHT", "SPO-MIGHT"]
data_means = [
    [0.660, 0.540, 0.697, 0.628, 0.657],  # Cohort 1
    [0.380, 0.296, 0.398, 0.331, 0.366]   # Cohort 2
]
data_devs = [
    [0.028, 0.048, 0.019, 0.038, 0.041],  # Cohort 1
    [0.025, 0.031, 0.017, 0.022, 0.029]   # Cohort 2
]

x = np.arange(len(cohorts))  
width = 0.15  
colors = ['b', 'g', 'r', 'orange', 'purple']  


fig, ax = plt.subplots(figsize=(12, 8))


for i, model in enumerate(models):
    ax.bar(
        x + i * width - (width * len(models) / 2), 
        [data_means[0][i], data_means[1][i]],      
        width,
        yerr=[data_devs[0][i], data_devs[1][i]],    
        color=colors[i],                            
        alpha=0.8,
        label=model,                                
        capsize=5
    )

ax.set_xlabel("Cohorts", fontsize=22)
ax.set_ylabel("S@98 (Mean ± Deviation)", fontsize=22)
ax.set_title("S@98 Performance Comparison (5000 Trees)", fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(cohorts, fontsize=22)


ax.legend(title="Methods", fontsize=22, title_fontsize=22)


ax.grid(alpha=0.3)


output_file = "./figures/sporf_related_s98_5000.png"
os.makedirs("./figures", exist_ok=True)
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Bar chart saved at: {output_file}")
