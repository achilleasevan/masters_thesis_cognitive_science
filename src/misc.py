import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import shapiro, levene, kruskal
from compare_params_of_representations import load_time_series
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt


# def class_statistics():
#     # Define constants
#     fs = 15385  # Sampling frequency in Hz
#     root_folder = r"D:\1.Thesis\1.Data_Analysis\data_noevents"
#     output_csv = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\class_statistics.csv"

#     # Prepare a list to collect statistics
#     class_statistics = []

#     # Iterate through each class folder
#     for subfolder in ['control', 'ictal', 'postSD', 'preSD']:
#         subfolder_path = os.path.join(root_folder, subfolder)

#         if not os.path.exists(subfolder_path):
#             print(f"Warning: Subfolder '{subfolder}' does not exist.")
#             continue

#         max_time = -np.inf
#         min_time = np.inf
#         all_means = []
#         all_std_devs = []

#         # Iterate through `.mat` files in the current subdirectory
#         for root, _, files in os.walk(subfolder_path):
#             for file in files:
#                 if file.endswith('.mat'):
#                     file_path = os.path.join(root, file)
#                     try:
#                         # Load the `.mat` file
#                         data = loadmat(file_path)

#                         # Filter for the data key (ignoring metadata keys)
#                         filtered_keys = [key for key in data.keys() if not key.startswith('__')]
#                         if not filtered_keys:
#                             print(f"No valid data key found in {file_path}")
#                             continue

#                         filtered_key = filtered_keys[0]
#                         raw_series = data[filtered_key][0]
#                         num_samples = len(raw_series)

#                         # Construct time axis
#                         time_axis = np.arange(num_samples) / fs  # Time in seconds

#                         # Update min and max time values
#                         min_time = min(min_time, time_axis[-1])
#                         max_time = max(max_time, time_axis[-1])

#                         # Compute statistics
#                         all_means.append(np.mean(raw_series))
#                         all_std_devs.append(np.std(raw_series))

#                     except (OSError, KeyError, IndexError, ValueError) as e:
#                         print(f"Error processing file '{file_path}': {e}")
#                         continue

#         # Compute global statistics for the class
#         class_mean = np.mean(all_means) if all_means else None
#         class_std = np.mean(all_std_devs) if all_std_devs else None

#         # Store the results
#         class_statistics.append({
#             "Class": subfolder,
#             "Min Time (s)": min_time,
#             "Max Time (s)": max_time,
#             "Mean Amplitude": class_mean,
#             "Std Dev Amplitude": class_std
#         })

#     # Convert to DataFrame and save to CSV
#     df_stats = pd.DataFrame(class_statistics)
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df_stats.to_csv(output_csv, index=False)

#     print(f"Class statistics saved to {output_csv}")
    
"""
Statistical test comparing mean amplitudes for each class
"""

def amplitudes_of_classes():
    # Define constants
    root_folder = r"D:\1.Thesis\1.Data_Analysis\data_noevents"
    amplitude_csv = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\mean_amplitudes_per_series.csv"

    # Prepare list to collect per-time-series means
    all_mean_amplitudes = []

    for subfolder in ['control', 'ictal', 'postSD', 'preSD']:
        subfolder_path = os.path.join(root_folder, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"Warning: Subfolder '{subfolder}' does not exist.")
            continue

        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    try:
                        data = loadmat(file_path)
                        filtered_keys = [key for key in data.keys() if not key.startswith('__')]
                        if not filtered_keys:
                            print(f"No valid data key found in {file_path}")
                            continue

                        filtered_key = filtered_keys[0]
                        raw_series = data[filtered_key][0]

                        mean_amplitude = np.mean(raw_series)

                        all_mean_amplitudes.append({
                            "Class": subfolder,
                            "Mean Amplitude": mean_amplitude
                        })

                    except Exception as e:
                        print(f"Error processing file '{file_path}': {e}")
                        continue

    # Save all individual mean amplitudes
    df_means = pd.DataFrame(all_mean_amplitudes)
    os.makedirs(os.path.dirname(amplitude_csv), exist_ok=True)
    df_means.to_csv(amplitude_csv, index=False)
    
# amplitudes_of_classes()

# File paths
input_csv = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\mean_amplitudes_per_series.csv"
output_txt = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\normality_test.txt"
output_txt_kruskal_wallis = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\kruskal_wallis_result.txt"
output_txt_dunns = r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\dunns_result.txt"

# # Load the data
# df = pd.read_csv(input_csv)

# # Group by class
# grouped = df.groupby("Class")["Mean Amplitude"]

# # Run Shapiro-Wilk test for normality per group
# shapiro_results = {}
# for name, group in grouped:
#     if len(group) >= 3:  # Shapiro requires at least 3 data points
#         stat, p = shapiro(group)
#         shapiro_results[name] = (stat, p)
#     else:
#         shapiro_results[name] = ("Insufficient data", "N/A")

# # Run Levene's test for equal variances across all groups
# groups = [group.tolist() for _, group in grouped]
# levene_stat, levene_p = levene(*groups)

# # Save results to text file
# with open(output_txt, "w") as f:
#     f.write("Shapiro-Wilk Normality Test Results (per class):\n")
#     for cls, result in shapiro_results.items():
#         f.write(f"{cls}: Statistic={result[0]}, p-value={result[1]}\n")

#     f.write("\nLevene’s Test for Equal Variances:\n")
#     f.write(f"Statistic={levene_stat}, p-value={levene_p}\n")

# output_txt

def run_kruskal_wallis_test(input_csv, output_txt):
    """
    Runs the Kruskal-Wallis test on mean amplitudes grouped by class.
    
    Parameters:
    - input_csv (str): Path to CSV file containing 'Class' and 'Mean Amplitude' columns.
    - output_txt (str): Path to save the test results as a text file.
    """
    # Load the data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File not found at {input_csv}")
        return
    
    # Group the data by 'Class'
    grouped = df.groupby("Class")["Mean Amplitude"]
    group_data = [group.tolist() for _, group in grouped]

    # Run Kruskal-Wallis test
    stat, p = kruskal(*group_data)

    # Prepare results text
    result_text = (
        "Kruskal-Wallis H-test for Mean Amplitudes Across Classes\n"
        "----------------------------------------------------------\n"
        f"Statistic: {stat:.4f}\n"
        f"p-value  : {p:.4e}\n"
    )

    # Save to output file
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w') as f:
        f.write(result_text)

    print(f"Kruskal-Wallis test results saved to:\n{output_txt}")
    
# run_kruskal_wallis_test(
# input_csv=input_csv,
# output_txt=output_txt_kruskal_wallis
# )

def run_dunns_test(input_csv, output_txt, correction_method='bonferroni'):
    """
    Performs Dunn's test with correction on grouped data from a CSV.
    
    Parameters:
    - input_csv (str): Path to the CSV file with 'Class' and 'Mean Amplitude' columns.
    - output_txt (str): File path to save the results as a text file.
    - correction_method (str): Method for p-value adjustment ('bonferroni', 'holm', etc.)
    """
    # Load the data
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File not found at {input_csv}")
        return

    # Run Dunn's test
    dunn_result = sp.posthoc_dunn(df, val_col='Mean Amplitude', group_col='Class', p_adjust=correction_method)

    # Format output
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w') as f:
        f.write(f"Dunn's Test Results (p-values adjusted with {correction_method.title()} correction)\n")
        f.write("-" * 60 + "\n")
        f.write(dunn_result.to_string())
    
    print(f"Dunn's test results saved to:\n{output_txt}")

# run_dunns_test(
#     input_csv=r"D:\1.Thesis\1.Data_Analysis\experiments\results\class_info\mean_amplitudes_per_series.csv",
#     output_txt=output_txt_dunns,
#     correction_method='holm'
# )   
    
    
"""
Random plot
"""

# fs = 15385
# file_path = r"D:\1.Thesis\1.Data_Analysis\data_noevents\ictal\satb1__1376.7_fifth 20 min in 0 Mg_channel1.mat_noevent_after_last"
# time_series = load_time_series(file_path=file_path, downsampling=None)
# num_samples = len(time_series)

# # Construct time axis
# time_axis = np.arange(num_samples) / fs

# # Plot in high definition
# plt.figure(figsize=(12, 6), dpi=300)
# plt.plot(time_axis, time_series, color='blue', linewidth=1)
# plt.xlabel("Time (seconds)", fontsize=14, fontweight="bold")
# plt.ylabel("Amplitude", fontsize=14, fontweight="bold")
# plt.title("Quiescent Segment Ictal", fontsize=16, fontweight="bold")
# plt.grid(True, linestyle='--', alpha=0.6)

# # Save the plot
# output_path = r"D:\1.Thesis\1.Data_Analysis\experiments\results\plots\class_plots\ictal.png"
# plt.savefig(output_path, dpi=300, bbox_inches="tight")

# # # Show the plot
# # plt.show()

# # Provide the file path for download
# output_path