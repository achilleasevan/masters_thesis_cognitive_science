import os
import numpy as np
import matplotlib.pyplot as plt
from compare_params_of_representations import load_time_series
import pywt

def plot_graph_bins_on_timeseries():
    ROOT_PATH = r"D:\1.Thesis\1.Data_Analysis\data_noevents\control\satb1_1376.7_control_channel1.mat_noevent_between_6_7.mat"
    bin_edges_path = r"D:\1.Thesis\1.Data_Analysis\experiments\n_gram\bin_edges.npy"
    time_series = load_time_series(file_path=ROOT_PATH, downsampling=None)
    output_path = r"D:\1.Thesis\1.Data_Analysis\experiments\results\plots\graph_binning"
    os.makedirs(output_path, exist_ok=True)

    bin_edges = np.load(bin_edges_path)
    time_series = load_time_series(file_path=ROOT_PATH, downsampling=None)

    # Parameters
    chunk_size = 6000  # Number of samples per chunk
    num_bins = 10  # Number of bins for the y-axis

    # Step 1: Create bin edges and assign letters
    bin_edges[0] = np.min(time_series)
    bin_edges[-1] = np.max(time_series)
    letters = [chr(i) for i in range(ord('A'), ord('A') + num_bins)]
    bin_mapping = {letters[i]: (bin_edges[i], bin_edges[i + 1])
                for i in range(len(letters))}

    # Step 2: Divide the x-axis into chunks
    n_chunks = len(time_series) // chunk_size
    chunks = np.array_split(time_series[:n_chunks * chunk_size], n_chunks)

    # Internal function to assign letters to chunk means
    def assign_letter(value, bin_mapping):
        for letter, (low, high) in bin_mapping.items():
            if low <= value < high:
                return letter
        if value == bin_edges[-1]:
            return letters[-1]
        return None

    # Compute the mean of each chunk and assign a letter
    chunk_means = np.array([np.mean(chunk) for chunk in chunks])
    chunk_symbols = [assign_letter(mean, bin_mapping) for mean in chunk_means]

    # Plot the time-series
    plt.figure(figsize=(14, 7))
    plt.plot(time_series, label="Quiescence Segment", color="blue", linewidth=1)

    # Step 1: Plot chunk means
    for idx, (chunk_start, chunk_end, mean_value) in enumerate(
            zip(range(0, len(time_series), chunk_size), 
                range(chunk_size, len(time_series) + chunk_size, chunk_size), 
                chunk_means)):
        plt.hlines(mean_value, chunk_start, min(chunk_end, len(time_series)),
                colors='orange', linestyles='-', linewidth=1.5, label='Chunk Mean' if idx == 0 else "")


    # Step 2: Visualize the y-axis bins
    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        plt.axhline(y=low, color="gray", linestyle="--", linewidth=0.5)
        plt.text(len(time_series) + 50, (low + high) / 2,
                letters[i], va="center", ha="left", color="red",
                fontsize=12, fontweight="bold")

    # Add labels, legend, and grid
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Segment with Chunk Means and Y-Axis Bins", fontsize=14)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save the plot in high resolution
    output_file_with_chunks = os.path.join(output_path, "graph_binning_with_chunks_high_res1.png")
    plt.savefig(output_file_with_chunks, dpi=300, bbox_inches="tight")
    plt.close()

    output_file_with_chunks

# plot_graph_bins_on_timeseries()


def plot_selected_wavelets():
    # Specify the wavelets to be plotted
    wavelets = ["cmor0.5-0.5", "cmor1.5-1.0", "cmor2.5-1.5"]

    # Output directory
    output_dir = r"D:\1.Thesis\1.Data_Analysis\experiments\results\plots\wavelets"
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # Plot each specified wavelet
    for ax, wavelet in zip(axs, wavelets):
        [psi, x] = pywt.ContinuousWavelet(wavelet).wavefun(10)
        ax.plot(x, np.real(psi), color="blue")
        ax.plot(x, np.imag(psi), color="orange")
        ax.set_title(wavelet)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-0.8, 1])

    # Add a single legend for the entire figure
    lines = [plt.Line2D([0], [0], color="blue", label="Real Part"),
            plt.Line2D([0], [0], color="orange", label="Imaginary Part")]
    fig.legend(handles=lines, loc="upper right", ncol=2, fontsize=12)

    # Add a title for the entire figure
    plt.suptitle("Selected Complex Morlet Wavelets", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save the plot in high resolution
    output_file = os.path.join(output_dir, "selected_wavelets_high_res.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Wavelet plots saved at: {output_file}")