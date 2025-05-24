import pandas as pd
import numpy as np
from scipy.io import loadmat
import scipy.signal as sn
import pickle
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn import neighbors
from statsmodels.tsa.api import AutoReg
from file_paths import *

"""N-Gram Graphs Parameter Configurations"""
def compare_n_gram_params():
    # Load the results
    csv_path = N_GRAM_PARAMS_RESULTS_CSV_PATH
    results_df = pd.read_csv(csv_path)

    # Group by chunk size and representation, then calculate mean accuracy and F1 macro
    grouped_results = results_df.groupby(["representation", "chunk_size"]).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_f1_macro=("f1_macro", "mean")
    ).reset_index()

    best_accuracy = grouped_results.sort_values(by="mean_accuracy", ascending=False).head(1)
    best_f1_macro = grouped_results.sort_values(by="mean_f1_macro", ascending=False).head(1)

    print("Best Configuration by Accuracy:")
    print(best_accuracy)

    print("\nBest Configuration by F1 Macro:")
    print(best_f1_macro)

    comparison_csv_path = N_GRAM_STATISTICAL_COMPARISON_OF_PARAMS_CSV_PATH
    grouped_results.to_csv(comparison_csv_path, index=False)
    print(f"Comparison results saved to {comparison_csv_path}")
 
def load_time_series(file_path, downsampling):
    data = loadmat(file_path)
    filtered_keys = [key for key in data.keys() if not key.startswith('__')]
    if not filtered_keys:
        print(f"No valid keys found in: {file_path}")
    filtered_key = filtered_keys[0]
    raw_series = data[filtered_key][0]
    if downsampling is not None:
        raw_series = sn.decimate(
            raw_series, q=downsampling, axis=0)
    return raw_series

"""Wavelet Parameter Configurations"""
def visually_inspect_frequencies():
    # Load the saved folds
    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)

    # Extract data
    splits = folds_file["splits"]
    all_files = np.array(folds_file["all_files"])  # File paths
    all_labels = np.array(folds_file["all_labels"])  # Class labels
    # Define sampling frequency
    fs = 15385  # Sampling frequency

    # Select 20% of the files as a representative subset
    subset_size = int(0.2 * len(all_files))  # Select 20% of the data
    subset_indices = np.random.choice(len(all_files), size=subset_size, replace=False)
    subset_files = all_files[subset_indices]
    subset_labels = all_labels[subset_indices]

    subset_time_series = []
    for file_path in subset_files:
        try:
            time_series = load_time_series(file_path, downsampling=10)
            subset_time_series.append(time_series)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Perform FFT on the subset and plot
    for i, (time_series, label) in enumerate(zip(subset_time_series, subset_labels)):
        n = len(time_series)
        freq = fftfreq(n, d=1 / fs)[:n // 2]  # Frequency bins
        fft_values = np.abs(fft(time_series))[:n // 2]  # FFT magnitudes

        # Plot the frequency spectrum
        plt.plot(freq, fft_values, label=f"Label: {label} (Series {i + 1})")
        plt.title(f"Frequency Spectrum - Series {i + 1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, 500)  # Focus on relevant frequencies
        plt.legend()
        plt.show()
        
def extract_represented_data_wavelets(file_names, file_labels, downsampling, save_name, frequencies, wavelet):
    """ Recursively scan file_names and process .mat files. """
    ml_matrix = []
    # representative_graphs_folder = os.path.normpath(input("Please provide the representative graphs folder path (str): "))
    
    file_to_index = {name: idx for idx, name in enumerate(file_names)}
    
    for file in tqdm(file_names):
            try:
                data = loadmat(file)
                filtered_keys = [key for key in data.keys() if not key.startswith('__')]
                if not filtered_keys:
                    print(f"No valid keys found in: {file}")
                    continue
                filtered_key = filtered_keys[0]
                raw_series = data[filtered_key][0]
                if downsampling is not None:
                    raw_series = sn.decimate(
                        raw_series, q=downsampling, axis=0)
                representation = wavelet_repr(
                    time_series=raw_series, 
                    frequencies=frequencies, 
                    wavelet=wavelet
                    )
                # Attach class label
                # Δίνεις το όνομα του αρχείου και σου δίνει τον δείκτη του
                index = file_to_index[file]
                class_num = file_labels[index]
                representation = np.append(representation, class_num)
                
                ml_matrix.append(representation)
                
            except (OSError, ValueError, KeyError, IndexError) as e:
                print(f"Failed to process: {file} - Error: {e}")
                continue
            
    ml_matrix = np.vstack(ml_matrix)
    if save_name is not None:
        np.save(save_name, ml_matrix)
    return ml_matrix


def wavelet_repr(time_series, frequencies, wavelet):
    fs = 15385
    widths = pywt.frequency2scale(wavelet=wavelet, freq=frequencies) * fs
    coefs, _ = pywt.cwt(time_series, scales=widths,
                        wavelet=wavelet, sampling_period=1/fs)
    coefs = np.mean(coefs, axis=1)
    return coefs


def extract_ml_matrices_for_different_wavelet_configs():
    frequencies = [[2, 5, 10, 20, 40, 60], [2, 4, 6, 8, 10, 15], [5, 15, 30, 50, 70, 100]]
    wavelets = ["cmor0.5-0.5", "cmor1.5-1.0", "cmor1.5-1.5"]
    # Load saved splits
    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)

    all_files = np.array(folds_file["all_files"])
    all_labels = np.array(folds_file["all_labels"])

    # Iterate through combinations of frequencies and wavelets
    for frequency_set in frequencies:
        for wvlt in wavelets:
            downsampling = 10
            repr_length = 10
            
            # Create a descriptive filename
            frequency_str = "_".join(map(str, frequency_set))
            save_all_name = fr"D:\1.Thesis\1.Data_Analysis\experiments\ml_matrices\ml_matrix_wavelets_{frequency_str}_{wvlt}.npy"
            
            # Check if the matrix already exists
            if not os.path.exists(save_all_name):
                print(f"Processing: Frequencies={frequency_set}, Wavelet={wvlt}")
                
                # Generate the ML matrix
                ml_matrix = extract_represented_data_wavelets(
                    file_names=all_files,
                    file_labels=all_labels,
                    repr_name="wavelets",
                    repr_length=repr_length,
                    downsampling=downsampling,
                    save_name=save_all_name,
                    frequencies=frequency_set,
                    wavelet=wvlt
                )
                print(f"Saved ML matrix to {save_all_name}")
            else:
                print(f"ML matrix already exists: {save_all_name}")
            

def produce_results_for_different_wavelet_configs():
    repr_name = "wavelets"
    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)

    splits = folds_file["splits"]
    
    matrices_path = ML_MATRICES_PATH
    results_directory = WAVELET_PARAMS_RESULTS_DIRECTORY_PATH
    os.makedirs(results_directory, exist_ok=True)  # Ensure results directory exists

    for root, _, files in os.walk(matrices_path):
        # I have save 2 folders, "other" and "ready", in which I keep the discarded ml matrices. With this I skip them
        if any(folder in root for folder in ["other", "ready"]):
            continue
        for file in files:
            if file.endswith('.npy'):
                matrix_path = os.path.join(root, file)
                ml_matrix = np.load(matrix_path)

                # Extract wavelet and frequency information from the file name
                file_name = os.path.splitext(file)[0]  # Remove .npy extension
                wavelet_info = file_name.replace("ml_matrix_", "")  # Remove the prefix

                # Cross-validation using saved splits
                fold_results = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    x_train = ml_matrix[train_idx, :-1]
                    y_train = ml_matrix[train_idx, -1]
                    x_test = ml_matrix[test_idx, :-1]
                    y_test = ml_matrix[test_idx, -1]

                    x_train, y_train, x_test, y_test = np.abs(x_train), np.abs(y_train), np.abs(x_test), np.abs(y_test)
                    # Train KNN model
                    knn_model = neighbors.KNeighborsClassifier()
                    knn_model.fit(x_train, y_train)
                    y_pred = knn_model.predict(x_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_macro = f1_score(y_test, y_pred, average="macro")

                    fold_results.append({
                        "fold": fold_idx + 1,
                        "representation": repr_name,
                        "wavelet_info": wavelet_info,
                        "accuracy": accuracy,
                        "f1_macro": f1_macro
                    })

                    print(f"File: {file}, Fold {fold_idx + 1}: Accuracy = {accuracy:.4f}, F1 Macro = {f1_macro:.4f}")

                # Save results for this matrix file
                csv_file_name = f"{wavelet_info}_results.csv"
                csv_file_path = os.path.join(results_directory, csv_file_name)
                df = pd.DataFrame(fold_results)
                df.to_csv(csv_file_path, index=False)
                print(f"Results saved to {csv_file_path}")

    print("Processing complete.")

def compare_wavelet_params():
    # Define the path to the directory containing the CSV files
    results_directory = WAVELET_PARAMS_RESULTS_DIRECTORY_PATH

    # Initialize an empty DataFrame to store combined results
    combined_results = pd.DataFrame()

    # Iterate over all CSV files in the results directory
    for root, _, files in os.walk(results_directory):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                # Load the CSV file
                results_df = pd.read_csv(csv_path)
                # Add the file name as an identifier (optional, for traceability)
                results_df["file_name"] = file
                # Append to the combined DataFrame
                combined_results = pd.concat([combined_results, results_df], ignore_index=True)

    # Group by wavelet info and calculate mean accuracy and F1 macro
    grouped_results = combined_results.groupby("wavelet_info").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_f1_macro=("f1_macro", "mean")
    ).reset_index()

    # Find the best configurations
    best_accuracy = grouped_results.sort_values(by="mean_accuracy", ascending=False).head(1)
    best_f1_macro = grouped_results.sort_values(by="mean_f1_macro", ascending=False).head(1)

    print("Best Configuration by Accuracy:")
    print(best_accuracy)

    print("\nBest Configuration by F1 Macro:")
    print(best_f1_macro)

    # Save the grouped results to a CSV file
    comparison_csv_path = os.path.join(results_directory, "wavelet_comparison_results.csv")
    grouped_results.to_csv(comparison_csv_path, index=False)
    print(f"Comparison results saved to {comparison_csv_path}")

"""Auto-regressive modelling Parameter Configurations"""
def auto_regr_repr(time_series, repr_length, trend):
    ar_model = AutoReg(endog=time_series, lags=repr_length, trend=trend)
    ar_trained_model = ar_model.fit()
    coefs = ar_trained_model.params
    return coefs

def extract_represented_data_autoregr(file_names, file_labels, repr_length, downsampling, save_name, trend):
    """ Recursively scan file_names and process .mat files. """
    ml_matrix = []
    
    file_to_index = {name: idx for idx, name in enumerate(file_names)}
    
    for file in tqdm(file_names):
            try:
                data = loadmat(file)
                filtered_keys = [key for key in data.keys() if not key.startswith('__')]
                if not filtered_keys:
                    print(f"No valid keys found in: {file}")
                    continue
                filtered_key = filtered_keys[0]
                raw_series = data[filtered_key][0]
                if downsampling is not None:
                    raw_series = sn.decimate(
                        raw_series, q=downsampling, axis=0)
                representation = auto_regr_repr(
                    time_series=raw_series, 
                    repr_length=repr_length, 
                    trend=trend
                    )
                # Attach class label
                # Δίνεις το όνομα του αρχείου και σου δίνει τον δείκτη του
                index = file_to_index[file]
                class_num = file_labels[index]
                representation = np.append(representation, class_num)
                
                ml_matrix.append(representation)
                
            except (OSError, ValueError, KeyError, IndexError) as e:
                print(f"Failed to process: {file} - Error: {e}")
                continue
            
    ml_matrix = np.vstack(ml_matrix)
    if save_name is not None:
        np.save(save_name, ml_matrix)
    return ml_matrix

def extract_ml_matrices_for_different_autoregr_configs():
    lags = [5, 10, 15, 20]
    trends = ["n", "c", "ct"]
    # Load saved splits
    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)

    all_files = np.array(folds_file["all_files"])
    all_labels = np.array(folds_file["all_labels"])

    # Iterate through combinations of lags and trends
    for lag in lags:
        for trend in trends:
            downsampling = 10
            
            # Create a descriptive filename
            lag_str = f"lag_{lag}"
            trend_str = f"trend_{trend}"
            file_name = f"ml_matrix_autoregr_{lag_str}_{trend_str}.npy"
            save_all_name = ML_MATRICES_DIR / file_name
            
            # Check if the matrix already exists
            if not os.path.exists(save_all_name):
                print(f"Processing: Lags={lag}, Trend={trend}")
                
                # Generate the ML matrix
                ml_matrix = extract_represented_data_autoregr(
                    file_names=all_files,
                    file_labels=all_labels,
                    repr_length=lag,
                    downsampling=downsampling,
                    save_name=save_all_name,
                    trend=trend
                )
                print(f"Saved ML matrix to {save_all_name}")
            else:
                print(f"ML matrix already exists: {save_all_name}")
                
def produce_results_for_different_autoregr_configs():
    repr_name = "auto_regr"
    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)

    splits = folds_file["splits"]
    
    matrices_path = ML_MATRICES_PATH
    results_directory = AUTO_REGR_PARAMS_RESULTS_DIRECTORY_PATH
    os.makedirs(results_directory, exist_ok=True)  # Ensure results directory exists

    for root, _, files in os.walk(matrices_path):
        # I have save 2 folders, "other" and "ready", in which I keep the discarded ml matrices. With this I skip them
        if any(folder in root for folder in ["other", "ready"]):
            continue
        for file in files:
            if file.endswith('.npy'):
                matrix_path = os.path.join(root, file)
                ml_matrix = np.load(matrix_path)

                # Extract auto_regr and frequency information from the file name
                file_name = os.path.splitext(file)[0]  # Remove .npy extension
                auto_regr_info = file_name.replace("ml_matrix_", "")  # Remove the prefix

                # Cross-validation using saved splits
                fold_results = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    x_train = ml_matrix[train_idx, :-1]
                    y_train = ml_matrix[train_idx, -1]
                    x_test = ml_matrix[test_idx, :-1]
                    y_test = ml_matrix[test_idx, -1]

                    # Train KNN model
                    knn_model = neighbors.KNeighborsClassifier()
                    knn_model.fit(x_train, y_train)
                    y_pred = knn_model.predict(x_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1_macro = f1_score(y_test, y_pred, average="macro")

                    fold_results.append({
                        "fold": fold_idx + 1,
                        "representation": repr_name,
                        "auto_regr_info": auto_regr_info,
                        "accuracy": accuracy,
                        "f1_macro": f1_macro
                    })

                    print(f"File: {file}, Fold {fold_idx + 1}: Accuracy = {accuracy:.4f}, F1 Macro = {f1_macro:.4f}")

                # Save results for this matrix file
                csv_file_name = f"{auto_regr_info}_results.csv"
                csv_file_path = os.path.join(results_directory, csv_file_name)
                df = pd.DataFrame(fold_results)
                df.to_csv(csv_file_path, index=False)
                print(f"Results saved to {csv_file_path}")

    print("Processing complete.")
    
def compare_autoregr_params():
    # Define the path to the directory containing the CSV files
    results_directory = AUTO_REGR_PARAMS_RESULTS_DIRECTORY_PATH

    # Initialize an empty DataFrame to store combined results
    combined_results = pd.DataFrame()

    # Iterate over all CSV files in the results directory
    for root, _, files in os.walk(results_directory):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                # Load the CSV file
                results_df = pd.read_csv(csv_path)
                # Add the file name as an identifier (optional, for traceability)
                results_df["file_name"] = file
                # Append to the combined DataFrame
                combined_results = pd.concat([combined_results, results_df], ignore_index=True)

    # Group by auto_regr info and calculate mean accuracy and F1 macro
    grouped_results = combined_results.groupby("auto_regr_info").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_f1_macro=("f1_macro", "mean")
    ).reset_index()

    # Find the best configurations
    best_accuracy = grouped_results.sort_values(by="mean_accuracy", ascending=False).head(1)
    best_f1_macro = grouped_results.sort_values(by="mean_f1_macro", ascending=False).head(1)

    print("Best Configuration by Accuracy:")
    print(best_accuracy)

    print("\nBest Configuration by F1 Macro:")
    print(best_f1_macro)

    # Save the grouped results to a CSV file
    comparison_csv_path = os.path.join(results_directory, "auto_regr_comparison_results.csv")
    grouped_results.to_csv(comparison_csv_path, index=False)
    print(f"Comparison results saved to {comparison_csv_path}")

