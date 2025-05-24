import numpy as np
from sklearn import neighbors, metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.tsa.api import AutoReg
import pywt
from scipy.io import loadmat
import os
import scipy.signal as sn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from n_gram_graphs import ngram_graphs_rep
import pickle
from tqdm import tqdm
from other_classifiers import basic_classifier, dummy_classifier
from file_paths import *


def perform_experiment():
    all_results = []

    folds_path = FOLDS_PATH
    with open(folds_path, "rb") as f:
        folds_file = pickle.load(f)
        
    splits = folds_file["splits"]
    
    repr_all_names = ["dummy", "basic", "wavelets", "auto_regr", "n_gram_graphs"]
    for repr_name in repr_all_names:
            if repr_name == "dummy":
                results = dummy_classifier(folds_file=folds_file, strategy="prior")
                for fold_idx, (accuracy, f1, cm) in enumerate(
                zip(results["accuracies"], results["f1_macro"], results["confusion_matrix"])):
                    all_results.append({
                        "representation": repr_name,
                        "fold": fold_idx + 1,
                        "accuracy": accuracy,
                        "f1_macro": f1,
                        "confusion_matrix": cm
                    })
            elif repr_name == "basic":
                class_means_file_path = CLASS_MEANS_JSON_PATH
                results = basic_classifier(class_means_file=class_means_file_path, folds_file=folds_file)
                for fold_idx, (accuracy, f1, cm) in enumerate(
                    zip(results["accuracies"], results["f1_macro"], results["confusion_matrix"])):
                    all_results.append({
                        "representation": repr_name,
                        "fold": fold_idx + 1,
                        "accuracy": accuracy,
                        "f1_macro": f1,
                        "confusion_matrix": cm
                    })
            else:
                ml_matrix = np.load(fr'{ML_MATRICES_PATH}\ready\ml_matrix_{
                    repr_name}.npy')

                # Cross-validation using saved splits
                results = []
                accuracies = []
                f1_scores = []
                confusion_matrices = []
                folds = []

                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    x_train = ml_matrix[train_idx, :-1]
                    y_train = ml_matrix[train_idx, -1]
                    x_test = ml_matrix[test_idx, :-1]
                    y_test = ml_matrix[test_idx, -1]

                    if repr_name == "wavelets":
                        x_train, y_train, x_test, y_test = np.abs(x_train), np.abs(y_train), np.abs(x_test), np.abs(y_test)
                    # Train KNN model
                    knn_model = neighbors.KNeighborsClassifier()
                    knn_model.fit(x_train, y_train)
                    y_pred = knn_model.predict(x_test)

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="macro")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Calculate metrics
                    accuracies.append(acc)
                    f1_scores.append(f1)
                    confusion_matrices.append(cm)
                    folds.append(fold_idx+1)

                    print(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f}, F1 Macro = {f1:.4f}")

                for fold_idx, (accuracy, f1, cm) in enumerate(zip(accuracies, f1_scores, confusion_matrices)):
                    all_results.append({
                        "representation": repr_name,
                        "fold": fold_idx + 1,
                        "accuracy": accuracy,
                        "f1_macro": f1,
                        "confusion_matrix": cm
                    })

    # Save all results to a single CSV file
    directory = RESULTS_DIR
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "all_results.csv")
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(file_path, index=False)
    print(f"All results saved to {file_path}")

    return all_results


def extract_represented_data(file_names, file_labels, repr_name, repr_length, downsampling, save_name, representative_graphs_folder, chunk_size):
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
                representation = repr_only_one(
                    time_series=raw_series, 
                    repr_name=repr_name, 
                    repr_length=repr_length, representative_graphs_folder=representative_graphs_folder,
                    chunk_size=chunk_size
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


def repr_only_one(time_series, repr_name, repr_length, representative_graphs_folder, chunk_size):
    if repr_name == "auto_regr":
        lags = repr_length
        ar_model = AutoReg(time_series, lags, trend='n')
        ar_trained_model = ar_model.fit()
        coefs = ar_trained_model.params
    elif repr_name == "wavelets":
        fs = 15385
        frequencies = [2, 5, 10, 20, 40, 60]
        widths = pywt.frequency2scale('cmor1.5-1.0', frequencies) * fs
        coefs, _ = pywt.cwt(time_series, scales=widths,
                            wavelet='cmor1.5-1.0', sampling_period=1/fs)
        coefs = np.mean(coefs, axis=1)
    elif repr_name == "n_gram_graphs":
        coefs = ngram_graphs_rep(time_series, representative_graphs_folder=representative_graphs_folder, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported representation method: {repr_name}")
    return coefs


def conf_mat_4states(y_test, y_pred, title=None):
    confusion_matrix = metrics.confusion_matrix(
        y_test, y_pred, labels=[0, 1, 2, 3])
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="inferno",
        cbar=True,
    )
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[
               'Control', 'Pre_SD', 'Post_SD', 'Ictal'])
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=[
               'Control', 'Pre_SD', 'Post_SD', 'Ictal'])
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.suptitle("Confusion Matrix")
    if title != 'None':
        plt.title(title)
    plt.tight_layout()
    plt.show()


def find_max_min_values(root_folder):

    # Iterate over the specified subdirectories
    for subfolder in ['control', 'ictal', 'postSD', 'preSD']:
        subfolder_path = os.path.join(root_folder, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"Warning: Subfolder '{subfolder}' does not exist.")
            continue

        # Iterate through `.mat` files in the current subdirectory
        max_value = -1000000
        min_value = 1000000
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    try:
                        # Load the `.mat` file
                        data = loadmat(file_path)

                        # Filter for the data key (ignoring metadata keys)
                        filtered_key = [
                            key for key in data.keys() if not key.startswith('__')
                        ][0]
                        raw_series = data[filtered_key][0]
                        # Find min and max values
                        this_max_value = np.max(raw_series)
                        if this_max_value > max_value:
                            max_value = this_max_value
                        this_min_value = np.min(raw_series)
                        if this_min_value < min_value:
                            min_value = this_min_value

                    except (OSError, KeyError, IndexError) as e:
                        print(f"Error processing file '{file_path}': {e}")
                        continue

    return max_value, min_value

perform_experiment()