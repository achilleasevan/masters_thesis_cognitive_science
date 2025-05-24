import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import shapiro
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import pingouin as pg
import pandas as pd
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from file_paths import *

def concatenate_csv_files(input_directory, output_file):
    """
    Concatenate all CSV files in the input directory into a single CSV file.

    Parameters:
        input_directory (str): Path to the directory containing the CSV files.
        output_file (str): Path to save the concatenated CSV file.
    """
    # Initialize an empty list to store DataFrames
    dataframes = []

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                # Read the CSV file and append to the list
                df = pd.read_csv(file_path)
                dataframes.append(df)

    # Concatenate all DataFrames
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        # Save the concatenated DataFrame to a CSV file
        concatenated_df.to_csv(output_file, index=False)
        print(f"Concatenated CSV file saved to {output_file}")
    else:
        print(f"No CSV files found in the directory: {input_directory}")

# Specify the input directory and output file
input_directory = WAVELET_PARAMS_RESULTS_DIRECTORY_PATH
output_file = WAVELET_PARAMS_COMPARISON_RESULTS
# Call the function
concatenate_csv_files(input_directory, output_file)

"""Auto-regr"""
results_ar = AUTOREGR_PARAMS_COMPARISON_RESULTS / "all_results_auto_regr.csv"
df = pd.read_csv(results_ar)
# Initialize dictionaries to store results by representation
accuracies = {}
f1_macro = {}
confusion_matrices = {}

# Iterate over unique representations
for method in df['auto_regr_info'].unique():
    method_df = df[df['auto_regr_info'] == method]
    
    # Save metrics for the method
    accuracies[method] = method_df['accuracy'].tolist()
    f1_macro[method] = method_df['f1_macro'].tolist()
    
acc_autoregr_lags_10_trend_ct = accuracies.get("autoregr_lags_10_trend_ct", [])
acc_autoregr_lags_10_trend_c = accuracies.get("autoregr_lags_10_trend_c", [])
acc_autoregr_lags_10_trend_n = accuracies.get("autoregr_lags_10_trend_n", [])
acc_autoregr_lags_15_trend_ct = accuracies.get("autoregr_lags_15_trend_ct", [])
acc_autoregr_lags_15_trend_c = accuracies.get("autoregr_lags_15_trend_c", [])
acc_autoregr_lags_15_trend_n = accuracies.get("autoregr_lags_15_trend_n", [])
acc_autoregr_lags_20_trend_ct = accuracies.get("autoregr_lags_20_trend_ct", [])
acc_autoregr_lags_20_trend_c = accuracies.get("autoregr_lags_20_trend_c", [])
acc_autoregr_lags_20_trend_n = accuracies.get("autoregr_lags_20_trend_n", [])
acc_autoregr_lags_5_trend_ct = accuracies.get("autoregr_lags_5_trend_ct", [])
acc_autoregr_lags_5_trend_c = accuracies.get("autoregr_lags_5_trend_c", [])
acc_autoregr_lags_5_trend_n = accuracies.get("autoregr_lags_5_trend_n", [])

f1_autoregr_lags_10_trend_ct = f1_macro.get("autoregr_lags_10_trend_ct", [])
f1_autoregr_lags_10_trend_c = f1_macro.get("autoregr_lags_10_trend_c", [])
f1_autoregr_lags_10_trend_n = f1_macro.get("autoregr_lags_10_trend_n", [])
f1_autoregr_lags_15_trend_ct = f1_macro.get("autoregr_lags_15_trend_ct", [])
f1_autoregr_lags_15_trend_c = f1_macro.get("autoregr_lags_15_trend_c", [])
f1_autoregr_lags_15_trend_n = f1_macro.get("autoregr_lags_15_trend_n", [])
f1_autoregr_lags_20_trend_ct = f1_macro.get("autoregr_lags_20_trend_ct", [])
f1_autoregr_lags_20_trend_c = f1_macro.get("autoregr_lags_20_trend_c", [])
f1_autoregr_lags_20_trend_n = f1_macro.get("autoregr_lags_20_trend_n", [])
f1_autoregr_lags_5_trend_ct = f1_macro.get("autoregr_lags_5_trend_ct", [])
f1_autoregr_lags_5_trend_c = f1_macro.get("autoregr_lags_5_trend_c", [])
f1_autoregr_lags_5_trend_n = f1_macro.get("autoregr_lags_5_trend_n", [])

# Perform ANOVA for accuracies
anova_acc = f_oneway(
    acc_autoregr_lags_10_trend_ct,
    acc_autoregr_lags_10_trend_c,
    acc_autoregr_lags_10_trend_n,
    acc_autoregr_lags_15_trend_ct,
    acc_autoregr_lags_15_trend_c,
    acc_autoregr_lags_15_trend_n,
    acc_autoregr_lags_20_trend_ct,
    acc_autoregr_lags_20_trend_c,
    acc_autoregr_lags_20_trend_n,
    acc_autoregr_lags_5_trend_ct,
    acc_autoregr_lags_5_trend_c,
    acc_autoregr_lags_5_trend_n
)

# Perform ANOVA for F1 Macro
anova_f1 = f_oneway(
    f1_autoregr_lags_10_trend_ct,
    f1_autoregr_lags_10_trend_c,
    f1_autoregr_lags_10_trend_n,
    f1_autoregr_lags_15_trend_ct,
    f1_autoregr_lags_15_trend_c,
    f1_autoregr_lags_15_trend_n,
    f1_autoregr_lags_20_trend_ct,
    f1_autoregr_lags_20_trend_c,
    f1_autoregr_lags_20_trend_n,
    f1_autoregr_lags_5_trend_ct,
    f1_autoregr_lags_5_trend_c,
    f1_autoregr_lags_5_trend_n
)

# Prepare results for saving
results_dir = AUTOREGR_PARAMS_COMPARISON_RESULTS
results_txt = os.path.join(results_dir, "anova_results.txt")

with open(results_txt, "w") as file:
    file.write("ANOVA Results\n")
    file.write("=====================\n")
    file.write(f"Accuracy ANOVA\nF-statistic: {anova_acc.statistic}\nP-value: {anova_acc.pvalue}\n")
    file.write("=====================\n")
    file.write(f"F1 Macro ANOVA\nF-statistic: {anova_f1.statistic}\nP-value: {anova_f1.pvalue}\n")

results_txt

"""----------------NGGs----------------"""

results_ngg = N_GRAM_GRAPHS_PARAMS_COMPARISON_RESULTS / "all_n_gram_graph_results.csv"
dfngg = pd.read_csv(results_ngg)
# Initialize dictionaries to store results by representation
accuracies_ngg = {}
f1_macro_ngg = {}

# Iterate over unique representations
for method_ngg in dfngg['chunk_size'].unique():
    method_df_ngg = dfngg[dfngg['chunk_size'] == method_ngg]
    
    # Save metrics for the method
    accuracies_ngg[method_ngg] = method_df_ngg['accuracy'].tolist()
    f1_macro_ngg[method_ngg] = method_df_ngg['f1_macro'].tolist()
    
acc_100_ngg = accuracies_ngg.get(100, [])
acc_500_ngg = accuracies_ngg.get(500, [])
acc_1000_ngg = accuracies_ngg.get(1000, [])
acc_1500_ngg = accuracies_ngg.get(1500, [])

f1_100_ngg = f1_macro_ngg.get(100, [])
f1_500_ngg = f1_macro_ngg.get(500, [])
f1_1000_ngg = f1_macro_ngg.get(1000, [])
f1_1500_ngg = f1_macro_ngg.get(1500, [])

# Perform ANOVA for accuracies
anova_acc_ngg = f_oneway(
    acc_100_ngg,
    acc_500_ngg,
    acc_1000_ngg,
    acc_1500_ngg
)

# Perform ANOVA for F1 Macro
anova_f1_ngg = f_oneway(
    f1_100_ngg,
    f1_500_ngg,
    f1_1000_ngg,
    f1_1500_ngg
)

# Prepare results for saving
results_dir_ngg = N_GRAM_GRAPHS_PARAMS_COMPARISON_RESULTS
results_txt_ngg = os.path.join(results_dir_ngg, "anova_results_ngg.txt")

with open(results_txt_ngg, "w") as file:
    file.write("ANOVA Results\n")
    file.write("=====================\n")
    file.write(f"Accuracy ANOVA\nF-statistic: {anova_acc_ngg.statistic}\nP-value: {anova_acc_ngg.pvalue}\n")
    file.write("=====================\n")
    file.write(f"F1 Macro ANOVA\nF-statistic: {anova_f1_ngg.statistic}\nP-value: {anova_f1_ngg.pvalue}\n")

results_txt_ngg

"""----------------Wavelets----------------"""

results_wav = WAVELET_PARAMS_COMPARISON_RESULTS/ "all_results_wavelets.csv"
dfwav = pd.read_csv(results_wav)
# Initialize dictionaries to store results by representation
accuracies_wav = {}
f1_macro_wav = {}

# Iterate over unique representations
for method_wav in dfwav['wavelet_info'].unique():
    method_df_wav = dfwav[dfwav['wavelet_info'] == method_wav]
    
    # Save metrics for the method
    accuracies_wav[method_wav] = method_df_wav['accuracy'].tolist()
    f1_macro_wav[method_wav] = method_df_wav['f1_macro'].tolist()
    
acc_wavelets_2_4_6_8_10_15_cmor05_05 = accuracies_wav.get("wavelets_2_4_6_8_10_15_cmor0.5-0.5", [])
acc_wavelets_2_4_6_8_10_15_cmor15_10 = accuracies_wav.get("wavelets_2_4_6_8_10_15_cmor1.5-1.0", [])
acc_wavelets_2_4_6_8_10_15_cmor15_15 = accuracies_wav.get("wavelets_2_4_6_8_10_15_cmor1.5-1.5", [])
acc_wavelets_2_5_10_20_40_60_cmor05_05 = accuracies_wav.get("wavelets_2_5_10_20_40_60_cmor0.5-0.5", [])
acc_wavelets_2_5_10_20_40_60_cmor15_10 = accuracies_wav.get("wavelets_2_5_10_20_40_60_cmor1.5-1.0", [])
acc_wavelets_2_5_10_20_40_60_cmor15_15 = accuracies_wav.get("wavelets_2_5_10_20_40_60_cmor1.5-1.5", [])
acc_wavelets_5_15_30_50_70_100_cmor05_05 = accuracies_wav.get("wavelets_5_15_30_50_70_100_cmor0.5-0.5", [])
acc_wavelets_5_15_30_50_70_100_cmor15_10 = accuracies_wav.get("wavelets_5_15_30_50_70_100_cmor1.5-1.0", [])
acc_wavelets_5_15_30_50_70_100_cmor15_15 = accuracies_wav.get("wavelets_5_15_30_50_70_100_cmor1.5-1.5", [])

f1_wavelets_2_4_6_8_10_15_cmor05_05 = f1_macro_wav.get("wavelets_2_4_6_8_10_15_cmor0.5-0.5", [])
f1_wavelets_2_4_6_8_10_15_cmor15_10 = f1_macro_wav.get("wavelets_2_4_6_8_10_15_cmor1.5-1.0", [])
f1_wavelets_2_4_6_8_10_15_cmor15_15 = f1_macro_wav.get("wavelets_2_4_6_8_10_15_cmor1.5-1.5", [])
f1_wavelets_2_5_10_20_40_60_cmor05_05 = f1_macro_wav.get("wavelets_2_5_10_20_40_60_cmor0.5-0.5", [])
f1_wavelets_2_5_10_20_40_60_cmor15_10 = f1_macro_wav.get("wavelets_2_5_10_20_40_60_cmor1.5-1.0", [])
f1_wavelets_2_5_10_20_40_60_cmor15_15 = f1_macro_wav.get("wavelets_2_5_10_20_40_60_cmor1.5-1.5", [])
f1_wavelets_5_15_30_50_70_100_cmor05_05 = f1_macro_wav.get("wavelets_5_15_30_50_70_100_cmor0.5-0.5", [])
f1_wavelets_5_15_30_50_70_100_cmor15_10 = f1_macro_wav.get("wavelets_5_15_30_50_70_100_cmor1.5-1.0", [])
f1_wavelets_5_15_30_50_70_100_cmor15_15 = f1_macro_wav.get("wavelets_5_15_30_50_70_100_cmor1.5-1.5", [])


# Perform ANOVA for accuracies
anova_acc_wav = f_oneway(
    acc_wavelets_2_4_6_8_10_15_cmor05_05,
    acc_wavelets_2_4_6_8_10_15_cmor15_10,
    acc_wavelets_2_4_6_8_10_15_cmor15_15,
    acc_wavelets_2_5_10_20_40_60_cmor05_05,
    acc_wavelets_2_5_10_20_40_60_cmor15_10,
    acc_wavelets_2_5_10_20_40_60_cmor15_15,
    acc_wavelets_5_15_30_50_70_100_cmor05_05,
    acc_wavelets_5_15_30_50_70_100_cmor15_10,
    acc_wavelets_5_15_30_50_70_100_cmor15_15
)

# Perform ANOVA for F1 Macro
anova_f1_wav = f_oneway(
    f1_wavelets_2_4_6_8_10_15_cmor05_05,
    f1_wavelets_2_4_6_8_10_15_cmor15_10,
    f1_wavelets_2_4_6_8_10_15_cmor15_15,
    f1_wavelets_2_5_10_20_40_60_cmor05_05,
    f1_wavelets_2_5_10_20_40_60_cmor15_10,
    f1_wavelets_2_5_10_20_40_60_cmor15_15,
    f1_wavelets_5_15_30_50_70_100_cmor05_05,
    f1_wavelets_5_15_30_50_70_100_cmor05_05,
    f1_wavelets_5_15_30_50_70_100_cmor15_10,
    f1_wavelets_5_15_30_50_70_100_cmor15_15
)

# Prepare results for saving
results_dir_wav = WAVELET_PARAMS_COMPARISON_RESULTS
results_txt_wav = os.path.join(results_dir_wav, "anova_results_wav.txt")

with open(results_txt_wav, "w") as file:
    file.write("ANOVA Results\n")
    file.write("=====================\n")
    file.write(f"Accuracy ANOVA\nF-statistic: {anova_acc_wav.statistic}\nP-value: {anova_acc_wav.pvalue}\n")
    file.write("=====================\n")
    file.write(f"F1 Macro ANOVA\nF-statistic: {anova_f1_wav.statistic}\nP-value: {anova_f1_wav.pvalue}\n")

results_txt_wav

"""Tukey's HSD"""

results_wav = WAVELET_PARAMS_COMPARISON_RESULTS / "all_results_wavelets.csv"
dfwav = pd.read_csv(results_wav)

# Initialize dictionaries to store results by representation
accuracies_wav = {}
f1_macro_wav = {}

# Iterate over unique representations
for method_wav in dfwav['wavelet_info'].unique():
    method_df_wav = dfwav[dfwav['wavelet_info'] == method_wav]
    accuracies_wav[method_wav] = method_df_wav['accuracy'].tolist()
    f1_macro_wav[method_wav] = method_df_wav['f1_macro'].tolist()

# Prepare data for Tukey's HSD test for F1 Macro scores
wavelet_labels = []
f1_values = []

# Flatten data and associate labels for each wavelet configuration
for method, values in f1_macro_wav.items():
    wavelet_labels.extend([method] * len(values))
    f1_values.extend(values)

# Convert to numpy arrays
wavelet_labels = np.array(wavelet_labels)
f1_values = np.array(f1_values)

# Perform Tukey's HSD test
tukey_results = pairwise_tukeyhsd(endog=f1_values, groups=wavelet_labels, alpha=0.05)

# Save the results to a file
results_dir_wav = WAVELET_PARAMS_COMPARISON_RESULTS
tukey_results_txt = os.path.join(results_dir_wav, "tukey_hsd_f1_macro.txt")

with open(tukey_results_txt, "w") as file:
    file.write(str(tukey_results))

tukey_results_txt