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
###

### Setup local virtual environment
```Bash
python3 -m venv .venv
source .venv/bin/activate # Linux
source .venv\Scripts\Activate # Windows
python -m pip install -r requirements.txt
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
