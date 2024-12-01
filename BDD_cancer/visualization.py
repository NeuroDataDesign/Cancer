import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
sample_list_file = "AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]

# Check unique values in the cohort column
print(sample_list["cohort"].unique())

# Count samples for each cohort
cohort1_count = len(sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"])
cohort2_count = len(sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"])
pon_count = len(sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"])

print(f"Cohort1: {cohort1_count}")
print(f"Cohort2: {cohort2_count}")
print(f"PanelOfNormals: {pon_count}")

# Prepare data for the pie chart
counts = [cohort1_count, cohort2_count, pon_count]
labels = ["Cohort1", "Cohort2", "PanelOfNormals"]

# Create the output directory if it does not exist
output_dir = "BDD_cancer/figures/visual"
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_dir, "cohort_distribution_pie_chart.png")

# Plot the pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    counts,
    labels=labels,
    autopct='%1.1f%%',  # Show percentages
    startangle=90,  # Rotate to start from top
    wedgeprops={"edgecolor": "black"},  # Add borders to the slices
)
plt.title("Cohort Distribution")

# Save the figure
plt.savefig(output_file, dpi=300)
plt.close()  # Close the figure to avoid displaying in interactive environments

print(f"Pie chart saved at: {output_file}")