import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import os

file_path = "WiseCondorX.Wise-1.csv"
data = pd.read_csv(file_path)

# Important features in cohort 1, cohort 2, and ALL
columns_to_plot = [
    "17:43000001-44000000",  # Important 1
    "17:20000001-21000000",  # Important 2
    "8:140000001-141000000", # Important 3
    "4:68000001-69000000",   # Important 4
    "2:98000001-99000000",   # Important 5
    "13:60000001-61000000"
]

output_dir = "./figures/visual"
# os.makedirs(output_dir, exist_ok=True)

for column in columns_to_plot:
    # Get the data for the positive and negative classes
    cancer_positive = data[data["Cancer Status"] == "Cancer"][column].dropna()
    cancer_negative = data[data["Cancer Status"] == "Healthy"][column].dropna()
    
    # Plot the distribution of the feature
    plt.figure(figsize=(8, 6))
    plt.hist(cancer_positive, bins=50, alpha=0.5, label="Cancer", density=True, color='r')
    plt.hist(cancer_negative, bins=50, alpha=0.5, label="Healthy", density=True, color='b')
    
    # Plot the density curves of positive and negative classes
    density_positive = gaussian_kde(cancer_positive)
    x_positive = np.linspace(min(cancer_positive), max(cancer_positive), 500)
    plt.plot(x_positive, density_positive(x_positive), color='r', linestyle='-', label="Cancer Density")
    density_negative = gaussian_kde(cancer_negative)
    x_negative = np.linspace(min(cancer_negative), max(cancer_negative), 500)
    plt.plot(x_negative, density_negative(x_negative), color='b', linestyle='-', label="Healthy Density")
    
    
    # Add labels and title
    plt.title(f"Distribution of {column}", fontsize=20)
    plt.xlabel("Normalized CG", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{column.replace(':', '_')}_distribution.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved distribution plot for {column} at: {output_file}")