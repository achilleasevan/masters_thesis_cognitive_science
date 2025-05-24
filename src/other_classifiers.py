import os
import numpy as np
from scipy.io import loadmat
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
import pickle
from compare_params_of_representations import load_time_series
from tqdm import tqdm
from scipy.signal import decimate
from file_paths import *

def find_mean_of_each_class(root_folder, output_file):
    """
    Calculate the mean of each class ('control', 'ictal', 'postSD', 'preSD')
    and save the results to a file.

    Parameters:
        root_folder (str): Path to the root folder containing subfolders for each class.
        output_file (str): Path to the file where results will be saved.

    Returns:
        dict: A dictionary with the mean values for each class.
    """
    # Initialize a dictionary to store means for each class
    class_means = {'control': [], 'ictal': [], 'postSD': [], 'preSD': []}

    # Iterate over the specified subdirectories
    for subfolder in ['control', 'ictal', 'postSD', 'preSD']:
        subfolder_path = os.path.join(root_folder, subfolder)

        if not os.path.exists(subfolder_path):
            print(f"Warning: Subfolder '{subfolder}' does not exist.")
            continue

        # Iterate through `.mat` files in the current subdirectory
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    try:
                        # Load the `.mat` file
                        data = loadmat(file_path)

                        # Filter for the data key (ignoring metadata keys)
                        filtered_keys = [key for key in data.keys() if not key.startswith('__')]
                        if not filtered_keys:
                            print(f"No valid keys found in file '{file_path}'.")
                            continue

                        filtered_key = filtered_keys[0]
                        raw_series = data[filtered_key][0]

                        # Calculate the mean of the data
                        this_mean = np.mean(raw_series)

                        # Append the mean to the respective class
                        class_means[subfolder].append(this_mean)

                    except (OSError, KeyError, IndexError, ValueError) as e:
                        print(f"Error processing file '{file_path}': {e}")
                        continue

    # Calculate the overall mean for each class
    overall_means = {key: np.mean(values) if values else None for key, values in class_means.items()}

    # Save the results to a file
    with open(output_file, 'w') as f:
        json.dump(overall_means, f, indent=4)

    return overall_means

"""Basic Classifier"""
def create_ml_matrix_basic(save_name):
    folds_file = FOLDS_PATH
    # Load splits and data
    with open(folds_file, "rb") as f:
        data = pickle.load(f)

    all_files = np.array(data["all_files"])
    all_labels = np.array(data["all_labels"])
    file_to_index = {name: idx for idx, name in enumerate(all_files)}
    ml_matrix = []
    for file in tqdm(all_files):
        time_series = load_time_series(file_path=file, downsampling=None)
        series_mean = np.mean(time_series)
        index = file_to_index[file]
        class_num = all_labels[index]
        mean_with_class = np.append(series_mean, class_num)
        ml_matrix.append(mean_with_class)
    ml_matrix = np.vstack(ml_matrix)
    if save_name is not None:
        np.save(save_name, ml_matrix)
    return ml_matrix

def basic_classifier(class_means_file, folds_file):
    """
    A basic classifier that uses class mean values to classify instances with saved splits.

    Parameters:
        file_names_train (list): List of file paths for training data.
        file_names_test (list): List of file paths for testing data.
        class_means_file (str): Path to the JSON file containing class means and labels.
        folds_file (str): Path to the file containing saved splits.

    Returns:
        dict: A dictionary containing accuracy, F1 macro scores, and confusion matrix.
    """
    # Load class means from the JSON file
    with open(class_means_file, "r") as f:
        class_means = json.load(f)

    splits = folds_file["splits"]
    all_files = np.array(folds_file["all_files"])
    all_labels = np.array(folds_file["all_labels"])

    # Convert class means to a NumPy array for easier processing
    class_labels = list(class_means.keys())
    class_mean_values = np.array([class_means[label] for label in class_labels])

    # Map textual labels to numerical values
    label_mapping = {"control": 0, "preSD": 1, "postSD": 2, "ictal": 3}

    def classify(ml_matrix):
        """Classify instances based on the closest class mean."""
        predictions = []
        for instance in ml_matrix:

            # Find the closest class mean
            closest_class_idx = np.argmin(np.abs(class_mean_values - instance))
            predictions.append(class_labels[closest_class_idx])

        predictions = [label_mapping[prediction] for prediction in predictions]
        return predictions

    # Perform cross-validation using saved splits
    accuracies = []
    f1_scores = []
    confusion_matrices = []
    results = []
    folds = []
    matrix_path = BASELINE_CLASSIFIER_DIRECTORY / "ml_matrix_basic.npy"
    ml_matrix = np.load(matrix_path)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        x_test = ml_matrix[test_idx, :-1]
        y_test = ml_matrix[test_idx, -1]

        test_predictions = np.array(classify(x_test))

        acc = accuracy_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions, average='macro')
        cm = confusion_matrix(y_test, test_predictions)

        accuracies.append(acc)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        folds.append(fold_idx+1)

        print(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f}, F1 Macro = {f1:.4f}")
    results = {
        "fold": folds,
        "representation": "basic",
        "accuracies": accuracies,
        "f1_macro": f1_scores,
        "confusion_matrix": confusion_matrices
    }
    return results

def dummy_classifier(folds_file, strategy):
    """
    A dummy classifier that works with pre-saved splits for cross-validation.

    Parameters:
        folds_file (str): Path to the file containing the saved splits.
        strategy (str): Strategy for the dummy classifier (e.g., 'most_frequent', 'stratified', 'uniform').

    Returns:
        dict: Dictionary containing accuracy, F1 macro scores, and confusion matrices for each fold.
    """
    splits = folds_file["splits"]

    accuracies = []
    f1_scores = []
    confusion_matrices = []
    folds = []
    ml_matrix = np.load(fr'D:\1.Thesis\1.Data_Analysis\experiments\baseline_classifier\dummy_classifier_ml_matrix\dummy_ml_matrix.npy')

    # Iterate over the splits
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        x_train = ml_matrix[train_idx, :-1]
        y_train = ml_matrix[train_idx, -1]
        x_test = ml_matrix[test_idx, :-1]
        y_test = ml_matrix[test_idx, -1]

        # Initialize and train the dummy classifier
        dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
        dummy_clf.fit(x_train, y_train)

        # Predict on the test set
        y_pred = dummy_clf.predict(x_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)
        folds.append(fold_idx+1)

        # Append metrics
        accuracies.append(acc)
        f1_scores.append(f1)
        confusion_matrices.append(cm)
        print(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f}, F1 Macro = {f1:.4f}")
        
    results = {
        "fold": folds,
        "representation": "dummy",
        "accuracies": accuracies,
        "f1_macro": f1_scores,
        "confusion_matrix": confusion_matrices
    }
    return results

def find_minimum_len_value_of_dataset():
    folds_file = FOLDS_PATH
    # Load splits and data
    with open(folds_file, "rb") as f:
        data = pickle.load(f)

    all_files = np.array(data["all_files"])
    min_len = float('inf')
    for file in tqdm(all_files):
        time_series = load_time_series(file_path=file, downsampling=None)
        len_series = len(time_series)
        min_len = min(min_len, len_series)
    return min_len
    
def create_ml_matrix_for_dummy_classifier(save_name):
    folds_file = FOLDS_PATH
    # Load splits and data
    with open(folds_file, "rb") as f:
        data = pickle.load(f)

    all_files = np.array(data["all_files"])
    all_labels = np.array(data["all_labels"])
    file_to_index = {name: idx for idx, name in enumerate(all_files)}
    ml_matrix = []
    for file in tqdm(all_files):
        time_series = load_time_series(file_path=file, downsampling=None)
        truncated_series = time_series[:15256]
        index = file_to_index[file]
        class_num = all_labels[index]
        truncated_series = np.append(truncated_series, class_num)
        
        ml_matrix.append(truncated_series)
    ml_matrix = np.vstack(ml_matrix)
    if save_name is not None:
        np.save(save_name, ml_matrix)
    return ml_matrix
    


if __name__ == "__main__":
    find_mean_of_each_class(root_folder=DATA_DIR, output_file=CLASS_MEANS_JSON_PATH)
    create_ml_matrix_basic(save_name= BASELINE_CLASSIFIER_DIRECTORY / "ml_matrix_basic.npy")
    ml_matrix_dummy_path = BASELINE_CLASSIFIER_DIRECTORY / "dummy_classifier_ml_matrix"
    os.makedirs(ml_matrix_dummy_path, exist_ok=True)
    create_ml_matrix_for_dummy_classifier(save_name=ml_matrix_dummy_path / "dummy_ml_matrix.npy")