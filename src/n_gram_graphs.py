from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from scipy.io import loadmat
import os
import scipy.signal as sn
import pickle
from pyinsect.documentModel.representations import DocumentNGramGraph
from pyinsect.collector.NGramGraphCollector import NGramGraphCollector
from pyinsect.documentModel.comparators.NGramGraphSimilarity import SimilaritySS, SimilarityVS, SimilarityNVS
import json
from file_paths import *

NUM_BINS = 10


def extract_array_of_all_data_for_binning(root_folder, downsampling, save_name):
    """
    Recursively scan folders, process .mat files, and concatenate data.

    Parameters:
        root_folder (str): Root directory to scan for .mat files.
        repr_name (str): Name for representation (not used here but can tag output).
        downsampling (int or None): Factor to downsample the data (None to skip).
        save_name (str): Filename to save the concatenated array (None to skip).

    Returns:
        np.ndarray: Concatenated array of all time series.
    """
    data_storage = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                try:
                    # Load the .mat file
                    data = loadmat(file_path)
                    filtered_key = [
                        key for key in data.keys() if not key.startswith('__')][0]
                    raw_series = data[filtered_key][0]

                    # Apply downsampling if specified
                    if downsampling != 'None':
                        raw_series = sn.decimate(
                            raw_series, q=downsampling, axis=0)

                    # Append to storage
                    data_storage.append(raw_series)
                except (OSError, ValueError):
                    print(f"Failed to process file: {file_path}")
                    continue

    # Concatenate all time series into a single array
    concatenated_data = np.concatenate(data_storage)

    # Save the concatenated array if a save name is provided
    if save_name != 'None':
        np.save(save_name, concatenated_data)

    return concatenated_data


def get_bin_edges(data, num_bins, bin_edges_save_name):
    est = KBinsDiscretizer(
        n_bins=num_bins, encode='ordinal', strategy='quantile')

    flattened_data = data.reshape(-1, 1)
    est.fit(flattened_data)

    bin_edges = est.bin_edges_[0]

    bin_edges[0] = -np.inf  # Extend the lowest bin to -infinity
    bin_edges[-1] = np.inf  # Extend the highest bin to +infinity

    np.save(bin_edges_save_name, bin_edges)
    return bin_edges


def turn_timeseries_into_symbolseries(time_series, bin_edges, chunk_size):
    num_bins = len(bin_edges) - 1
    letters = [chr(i) for i in range(ord('A'), ord('A') + num_bins)]
    bin_mapping = {letters[i]: (bin_edges[i], bin_edges[i + 1]) for i in range(num_bins)}

    def assign_letter(value):
        for letter, (low, high) in bin_mapping.items():
            if low <= value < high:
                return letter
        if value == bin_edges[-1]:
            return letters[-1]
        return None

    n_chunks = len(time_series) // chunk_size
    chunks = np.array_split(time_series[:n_chunks * chunk_size], n_chunks)
    chunk_means = [np.mean(chunk) for chunk in chunks]
    symbol_series = "".join([assign_letter(mean) for mean in chunk_means])
    return symbol_series

def turn_data_into_symbol_series(file_paths, bin_edges, chunk_size):
    symbol_data = []
    for path in file_paths:
        if path.endswith('.mat'):
            try:
                data = loadmat(path)
                filtered_key = [key for key in data.keys() if not key.startswith('__')][0]
                raw_series = data[filtered_key][0]
                symbol_series = turn_timeseries_into_symbolseries(
                    time_series=raw_series,
                    bin_edges=bin_edges,
                    chunk_size=chunk_size
                )
                symbol_data.append(symbol_series)
            except (OSError, ValueError) as e:
                print(f"Failed to process file: {path}. Error: {e}")
                continue
    return symbol_data

def generate_symbol_series_for_all_chunks(file_paths, bin_edges_path, output_base_dir, is_training):
    """
    Generates symbol series for chunk sizes and saves them to appropriately named subfolders.

    Parameters:
        file_paths (list): List of .mat file paths
        bin_edges_path (str): Path to .npy bin edges
        output_base_dir (str or Path): Where to save chunked subfolders
        is_training (bool): True for training set, False for validation set
    """
    bin_edges = np.load(bin_edges_path)
    chunk_sizes = [100, 500, 1000, 1500, 2000]

    for chunk_size in chunk_sizes:
        subfolder = os.path.join(output_base_dir, f"chunk_size_{chunk_size}")
        os.makedirs(subfolder, exist_ok=True)

        symbol_data = turn_data_into_symbol_series(file_paths, bin_edges, chunk_size)

        if is_training:
            filename = "symbol_data_training.pkl"
        else:
            filename = "symbol_data_validation.pkl"

        output_file = os.path.join(subfolder, filename)
        with open(output_file, "wb") as f:
            pickle.dump(symbol_data, f)

        print(f"Saved: {output_file}")


def create_graph_from_symbol_series(symbol_series):
    """
    Creates a graph from a given symbol-series.

    Parameters:
        symbol_series (str): Symbol series to be converted into a graph.

    Returns:
        networkx.DiGraph: The produced graph.
    """
    # Initialize the DocumentNGramGraph
    NGramGraphClass = DocumentNGramGraph.DocumentNGramGraph
    ngg = NGramGraphClass(n=3, Dwin=3, Data=symbol_series)
    return ngg


def process_symbol_series_to_graphs(root_folder, output_root_folder):
    """
    Processes each folder of symbol-series, converts them to graphs, and saves the output.

    Parameters:
        root_folder (str): Root directory containing symbol-series folders.
        output_root_folder (str): Root directory where graph output folders will be saved.
    """
    # Ensure the output root folder exists
    os.makedirs(output_root_folder, exist_ok=True)

    # Iterate through each subfolder in the root folder
    for folder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, folder_name)

        # Skip files; process only folders
        if os.path.isdir(subfolder_path):
            # Create corresponding output folder for graphs
            output_subfolder = os.path.join(
                output_root_folder, f"graphs_{folder_name}")
            os.makedirs(output_subfolder, exist_ok=True)

            # Process each .pkl file in the subfolder
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(subfolder_path, file_name)

                    # Load symbol-series from the Pickle file
                    with open(file_path, "rb") as f:
                        symbol_series = pickle.load(f)

                    graph_storage = []
                    for series in symbol_series:
                        # Create the graph
                        graph = create_graph_from_symbol_series(series)
                        graph_storage.append(graph)

                    # Define the output file name and save the graph
                    output_file_name = f"graphs_{
                        file_name.replace('symbol_data_', '')}"
                    output_file_path = os.path.join(
                        output_subfolder, output_file_name)

                    with open(output_file_path, "wb") as f:
                        pickle.dump(graph_storage, f)

                    print(f"Saved graphs for {
                          file_name} to {output_file_path}")



def compute_representative_graphs(base_dir, split_file, output_dir):
    """
    Computes representative graphs for each class from training graphs in specified folders.

    Parameters:
        base_dir (str): Directory containing folders with training graph .pkl files.
        split_file (str): Path to the JSON file containing graph file paths and labels.
        output_dir (str): Directory to save representative graphs for each class.
    """
    # Mapping of text labels to numerical values
    label_mapping = {"control": 0, "preSD": 1, "postSD": 2, "ictal": 3}
    
    # Folders to process
    folders = [
        "graphs_chunk_size_100", "graphs_chunk_size_500",
        "graphs_chunk_size_1000", "graphs_chunk_size_1500", "graphs_chunk_size_2000"
    ]
    
    # Load the train-validation split JSON file
    with open(split_file, "r") as f:
        split_data = json.load(f)
    
    train_labels = split_data["graph_train_labels"]
    
    # Iterate over each folder
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        graph_file_path = os.path.join(folder_path, "graphs_training.pkl")
        
        if not os.path.exists(graph_file_path):
            print(f"Training graphs not found in {folder_path}. Skipping...")
            continue
        
        # Load training graphs
        with open(graph_file_path, "rb") as f:
            training_graphs = pickle.load(f)
        
        # Group graphs by class
        class_graphs = {label: [] for label in label_mapping.values()}
        for idx, graph in enumerate(training_graphs):
            label_text = train_labels[idx]
            label_num = label_mapping[label_text]
            class_graphs[label_num].append(graph)
        
        # Compute representative graphs for each class
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        for class_label, graphs in class_graphs.items():
            collector = NGramGraphCollector()
            for graph in graphs:
                collector._add_graph(graph)
            representative_graph = collector._representative_graph
            
            # Save the representative graph
            output_file = os.path.join(output_folder, f"class_{class_label}_representative_graph.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(representative_graph, f)
            print(f"Saved representative graph for class {class_label} in {output_file}.")


def ngram_graphs_rep(time_series, representative_graphs_folder, chunk_size):
    bin_edges_path = PATH_FOR_FILE_WITH_BIN_EDGES_FOR_GRAPHS
    bin_edges = np.load(bin_edges_path)
    symbol_series = turn_timeseries_into_symbolseries(time_series=time_series, num_bins=NUM_BINS, bin_edges=bin_edges, chunk_size=chunk_size)[0]
    graph = create_graph_from_symbol_series(symbol_series=symbol_series)
        # Array to store similarity values
    similarities = []

    # Iterate through the 4 representative graphs
    for class_idx in range(4):
        rep_graph_path = f"{representative_graphs_folder}/class_{class_idx}_representative_graph.pkl"

        # Load the representative graph
        with open(rep_graph_path, "rb") as f:
            representative_graph = pickle.load(f)

        # Initialize similarity measures
        simss = SimilaritySS()
        simvs = SimilarityVS()
        simnvs = SimilarityNVS()
        
        # Compute similarities
        ss = simss.getSimilarityDouble(ngg1=graph, ngg2=representative_graph)
        vs = simvs.getSimilarityDouble(ngg1=graph, ngg2=representative_graph)
        nvs = simnvs.getSimilarityDouble(ngg1=graph, ngg2=representative_graph)

        # Append the results
        similarities.extend([ss, vs, nvs])

    # Convert to a numpy array and return
    return np.array(similarities)

if __name__ == "__main__":
    extract_array_of_all_data_for_binning(root_folder=DATA_DIR, downsampling=None, save_name=N_GRAM_GRAPHS_DIRECTORY)
    concatenated_array_of_all_data = N_GRAM_GRAPHS_DIRECTORY / "concatenated_data.npy"
    get_bin_edges(data=concatenated_array_of_all_data, num_bins=NUM_BINS, bin_edges_save_name= N_GRAM_GRAPHS_DIRECTORY / "bin_edges.npy")
    
    with open(TRAIN_VALIDATION_SPLIT_FOR_GRAPHS_JSON_PATH, "r") as f:
        splits = json.load(f)

        generate_symbol_series_for_all_chunks(
            file_paths=splits["graph_train_files"],
            bin_edges_path=PATH_FOR_FILE_WITH_BIN_EDGES_FOR_GRAPHS,
            output_base_dir=SYMBOL_SERIES_OUTPUT_DIRECTORY,
            is_training=True
        )

        generate_symbol_series_for_all_chunks(
            file_paths=splits["graph_validation_files"],
            bin_edges_path=PATH_FOR_FILE_WITH_BIN_EDGES_FOR_GRAPHS,
            output_base_dir=SYMBOL_SERIES_OUTPUT_DIRECTORY,
            is_training=False
        )
    # Create Graphs from symbol series for each chunk size
    root_folder = SYMBOL_SERIES_OUTPUT_DIRECTORY
    output_root_folder = GRAPHS_OUTPUT_DIRECTORY

    process_symbol_series_to_graphs(root_folder, output_root_folder)

    # Create Representative Graphs
    base_directory = GRAPHS_OUTPUT_DIRECTORY
    split_file_path = TRAIN_VALIDATION_SPLIT_FOR_GRAPHS_JSON_PATH
    output_directory = REPRESENTATIVE_GRAPHS_OUTPUT_DIRECTORY

    # Run the function
    compute_representative_graphs(base_directory, split_file_path, output_directory)
    
"""Optional code below, with which I produced plot for the thesis presentation"""    
# Produce graph plot for an example symbol series (or part of it)
# sym_ser = r"D:\1.Thesis\1.Data_Analysis\experiments\n_gram\symbol_series\chunk_size_500\symbol_data_training.pkl"
# with open(sym_ser, "rb") as f:
#     sym_ser = pickle.load(f)
# this_sym = sym_ser[5]
# this_sym = this_sym[:24]
# NGramGraphClass = DocumentNGramGraph.DocumentNGramGraph
# ngg = NGramGraphClass(n=3, Dwin=3, Data=this_sym)
# ngg.GraphDraw(wf=False)