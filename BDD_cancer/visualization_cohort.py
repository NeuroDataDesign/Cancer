import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
sample_list_file = "AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "Sample", "cohort"]

# Check unique values in the cohort column
print(sample_list["cohort"].unique())

# Count samples for each cohort
cohort1_count = len(sample_list[sample_list["cohort"] == "Cohort1"]["Sample"])
cohort2_count = len(sample_list[sample_list["cohort"] == "Cohort2"]["Sample"])
pon_count = len(sample_list[sample_list["cohort"] == "PanelOfNormals"]["Sample"])

print(f"Cohort1: {cohort1_count}")
print(f"Cohort2: {cohort2_count}")
print(f"PanelOfNormals: {pon_count}")

# Prepare data for the pie chart
counts = [cohort1_count, cohort2_count, pon_count]
labels = ["Cohort1", "Cohort2", "PanelOfNormals"]

# Create the output directory if it does not exist
output_dir = "./figures/visual"

# Define the output file path
output_file = os.path.join(output_dir, "cohort_distribution_pie_chart.png")

# # Plot the pie chart
# plt.figure(figsize=(8, 6))
# plt.pie(
#     counts,
#     labels=labels,
#     autopct='%1.1f%%',  # Show percentages
#     startangle=90,  # Rotate to start from top
#     wedgeprops={"edgecolor": "black"},  # Add borders to the slices
#     textprops={"fontsize": 16},  # Set the text size
# )
# plt.title("Cohort Distribution", fontsize=24)

# # Save the figure
# plt.savefig(output_file, dpi=300)
# plt.close()  # Close the figure to avoid displaying in interactive environments

# print(f"Pie chart saved at: {output_file}")

# Read WiseCondorX.Wise-1.csv and get the columns names
root = "./"
file = "WiseCondorX.Wise-1.csv"
detail_file = pd.read_csv(root + file)
merged_data = pd.merge(sample_list, detail_file, on="Sample", how="inner")
# print(detail_file["Stage"].unique())
# print(merged_data.head())
cohorts = merged_data["cohort"].unique()

# Create a dictionary to map the original stage values to the merged values
stage_mapping = {
    'IA': 'I', 'IA1': 'I', 'IA2': 'I', 'IA3': 'I', 'IB': 'I', 'IC': 'I',
    'IIA': 'II', 'IIB': 'II', 'IIC': 'II', 'II': 'II',
    'IIIA': 'III', 'IIIA1': 'III', 'IIIA2': 'III', 'IIIIA': 'III', 'IIIB': 'III', 'IIIC': 'III', 'III': 'III', 'pT3N2Mx': 'III',
    'IVA': 'IV', 'IVB': 'IV', 'IV': 'IV',
    'Normal': 'Normal',
    'Not given': 'Unknown',
    'nan': 'Unknown',
    'biopsy only at time of blood draw': 'Unknown',
    '0': '0',
    '0__TisN0M0': '0',
    '0___TisN0M0': '0',
    '0__Tis(2)N0M0': '0',
    'IIIV': 'Unknown'  # maybe wrong
}

# Use the dictionary to create a new column with the merged stage values
merged_data['Stage_Merged'] = merged_data['Stage'].map(stage_mapping).fillna('Unknown')
print(merged_data['Stage_Merged'].unique())
for cohort in cohorts:
    cohort_data = merged_data[merged_data["cohort"] == cohort]
    stage_counts = cohort_data["Stage_Merged"].value_counts()

    # Prepare data for the pie chart
    labels = stage_counts.index
    sizes = stage_counts.values
    explode = [0.1 if label == 'IV' else 0 for label in labels]  # Explode One slice
    colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555', '#ffa07a', '#7cfc00'][:len(labels)]
    
    # Print stage counts
    print(f"\nCancer Stage Distribution for {cohort}:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count}")
        
    def autopct_with_threshold(pct, all_values):
        """Return the percentage as a string if it is greater than 1, otherwise return an empty string."""
        absolute = int(round(pct / 100. * sum(all_values)))
        return f'{pct:.1f}%' if pct > 1 else ''
    
    # Plot the pie chart
    plt.figure(figsize=(10, 6))
    plt.axes(aspect='equal')
    wedges, texts, autotexts = plt.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda pct: autopct_with_threshold(pct, sizes),
        pctdistance=0.9,
        labeldistance=1.1,
        startangle=180,
        radius=1.0,
        counterclock=False,
        wedgeprops={'linewidth': 1.5, 'width': 0.5, 'edgecolor': 'green'},
        textprops={'fontsize': 16, 'color': 'k'},
        center=(0, 0),
        frame=False
    )
    
    # Add a legend
    plt.legend(
        wedges,  
        labels,  
        title="Stages",  
        loc="lower left", 
        bbox_to_anchor=(1, 0, 0.5, 1),  
        fontsize=18  
    )

    plt.title(f"Cancer Stage Distribution in {cohort}", fontsize=18)

    # Save plot to file
    output_file_cohort = os.path.join(output_dir, f"{cohort}_stage_distribution.png")
    plt.savefig(output_file_cohort, dpi=300)
    plt.close()

    print(f"Saved plot for {cohort} at: {output_file_cohort}")