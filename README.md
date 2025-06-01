# MSc Thesis - Cognitive Science - Classification of quiescent LFP segments in states of epilepsy
The master's thesis project for the MSc in Cognitive Science at NKUA. Comparison of LFP data representations for classification of quiescent segments in states of epilepsy.

# Setup
If python3.12 is not installed on your system install by
following the [Pyenv installation Guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)

### Install python version 3.12.3
```Bash
pyenv install 3.12.3
pyenv local 3.12.3
```

### Setup local virtual environment
```Bash
python3 -m venv .venv
source .venv/bin/activate # Linux
source .venv\Scripts\Activate # Windows
```

### Install Megatools
```Bash
sudo apt install megatools
```

### Download Dataset from MEGA store ###

```Bash
mkdir data
cd data
megadl https://mega.nz/folder/1zMzHKLC#qTtWgbgb0_7ic7w_bKM-Kg
```
Note: If the above link doesn't work, contact us at hdaystest@gmail.com .

### With Conda / Mamba
To recreate the full environment (recommended for Conda/Mamba users):

```Bash
mamba create --name myenv --file requirements-conda.txt
```
or
```Bash
conda create --name myenv --file requirements-conda.txt
```

### With Pip
To install Python packages only:

```Bash
pip install -r requirements.txt
```

# Script Descriptions and Usage

- **src/file_paths.py**
  Centralizes all file and directory paths used by the other scripts. Be sure to replace these with the appropriate paths on your own system. Also, ensure that any directories specified for saving output files already exist to avoid runtime errors.

  - **src/train_test_split.py**
  Creates the initial train–test split using the dataset file paths. Additionally, it generates a further train–validation split from the training set, specifically for use in the n_gram_graphs.py script to build representative graphs and help reduce overfitting.

 - **src/compare_params_of_representations.py**
   Runs stratified 10-fold cross-validation across different parameter configurations for each representation to identify the optimal setup for the final experiment.
The script saves the evaluation results as CSV files for later analysis.

- **src/n_gram_graphs.py**
   Implement the N-Gram-Graphs representation.

 - **src/statistics_for_parameters_of_each_representation.py**
   Runs statistical tests on the results produced by compare_params_of_representations.py to determine the optimal parameter configurations for each representation.

    - **src/other_classifiers.py**
   Implements the two naive baseline classifiers.

    - **src/main.py**
   Runs the main classification experiment.

    - **src/statistical_test.py**
   Perform statistical tests on the results of the main experiment to show which of the three representations performs best.

    - **src/graph_bins_plot.py**
   This script produces different plots that were used in the text and presentation of the thesis.

    - **src/misc.py**
   Miscellaneous script used for different purposes, most importantly to compare statistically the amplitudes of the different classes.

    - **src/animations.py**
   This script produces a short animation that was used in the presentation of the thesis, as a visual assistance when explaining the concept of autoregressive modeling.
