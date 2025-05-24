from pathlib import Path


"""
For the compare_params_of_representations.py file
"""

BASE_DIR = Path(r"D:\1.Thesis\1.Data_Analysis") # Base directory of the project

DATA_DIR = Path(r"D:\1.Thesis\1.Data_Analysis\data_noevents") # Directory of the quiescent segments data

"""
For the train_test_split.py file
"""
TRAIN_TEST_SPLIT_JSON_PATH = BASE_DIR / "experiments/train_test_split/train_test_split.json"

N_GRAM_PARAMS_RESULTS_CSV_PATH = BASE_DIR / "experiments/results/n_gram_graphs/all_n_gram_graph_results.csv"

N_GRAM_STATISTICAL_COMPARISON_OF_PARAMS_CSV_PATH = BASE_DIR / "experiments/results/n_gram_graphs/n_gram_graph_comparison_results.csv"

FOLDS_PATH = BASE_DIR / "experiments/train_test_split/folds.pkl"

ML_MATRICES_PATH = BASE_DIR / "experiments/ml_matrices"

WAVELET_PARAMS_RESULTS_DIRECTORY_PATH = BASE_DIR / "experiments/results/wavelets"

ML_MATRICES_DIR = BASE_DIR / "experiments" / "ml_matrices"

AUTO_REGR_PARAMS_RESULTS_DIRECTORY_PATH = BASE_DIR / "experiments/results/auto_regr"

"""For the statistics_for_parameters_of_each_representation.py file"""

RESULTS_PARAM_COMPARISON_DIRECTORY = BASE_DIR / "experiments" /"results_param_comparisons"

WAVELET_PARAMS_COMPARISON_RESULTS = RESULTS_PARAM_COMPARISON_DIRECTORY / "wavelets"
AUTOREGR_PARAMS_COMPARISON_RESULTS = RESULTS_PARAM_COMPARISON_DIRECTORY / "auto_regr"
N_GRAM_GRAPHS_PARAMS_COMPARISON_RESULTS = RESULTS_PARAM_COMPARISON_DIRECTORY / "ngg"

"""
For the compare_params_of_representations.py file
"""

"""For the n_gram_graphs.py file"""
TRAIN_VALIDATION_SPLIT_FOR_GRAPHS_JSON_PATH = BASE_DIR / "experiments/n_gram/train_validation_split/graph_train_validation_split.json"

PATH_FOR_FILE_WITH_BIN_EDGES_FOR_GRAPHS = BASE_DIR / "experiments/n_gram/bin_edges.npy"

SYMBOL_SERIES_OUTPUT_DIRECTORY = BASE_DIR / "experiments/n_gram/symbol_series"
N_GRAM_GRAPHS_DIRECTORY = BASE_DIR / "experiments/n_gram"
GRAPHS_OUTPUT_DIRECTORY = BASE_DIR / "experiments/n_gram/graphs"
REPRESENTATIVE_GRAPHS_OUTPUT_DIRECTORY = BASE_DIR / "experiments/n_gram/representative_graphs"

"""For the other_classifiers.py file"""
BASELINE_CLASSIFIER_DIRECTORY = BASE_DIR / "baseline_classifier"
CLASS_MEANS_JSON_PATH = BASELINE_CLASSIFIER_DIRECTORY / "class_means.json"

"""For main.py"""
RESULTS_DIR = BASE_DIR / "experiments/results"

"""For statistical_test.py"""
RESULTS_CSV_PATH = BASE_DIR / "experiments/results/all_results.csv"
RESULTS_TXT_PATH = BASE_DIR / "experiments/results/results_text"
PLOTS_PATH = BASE_DIR / "experiments/results/plots"
QQ_PLOTS_PATH = PLOTS_PATH / "QQ_plots"
BAR_PLOTS_PATH = PLOTS_PATH / "barplots"
CONFUSION_MATRICES_PATH = PLOTS_PATH / "confusion_matrices"