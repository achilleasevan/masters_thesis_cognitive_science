import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import shapiro
import numpy as np
from scipy.stats import friedmanchisquare, ranksums, wilcoxon
import pingouin as pg
import pandas as pd
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from file_paths import *

results_dir = RESULTS_CSV_PATH
df = pd.read_csv(results_dir)
# Initialize dictionaries to store results by representation
accuracies = {}
f1_macro = {}
confusion_matrices = {}

# Iterate over unique representations
for method in df['representation'].unique():
    method_df = df[df['representation'] == method]

    # Save metrics for the method
    accuracies[method] = method_df['accuracy'].tolist()
    f1_macro[method] = method_df['f1_macro'].tolist()
    confusion_matrices[method] = method_df['confusion_matrix'].tolist()

dummy_accuracies = accuracies.get("dummy", [])
basic_accuracies = accuracies.get("basic", [])
wavelets_accuracies = accuracies.get("wavelets", [])
auto_regr_accuracies = accuracies.get("auto_regr", [])
ngram_graphs_accuracies = accuracies.get("n_gram_graphs", [])

dummy_f1_macro = f1_macro.get("dummy", [])
basic_f1_macro = f1_macro.get("basic", [])
wavelets_f1_macro = f1_macro.get("wavelets", [])
auto_regr_f1_macro = f1_macro.get("auto_regr", [])
ngram_graphs_f1_macro = f1_macro.get("n_gram_graphs", [])

dummy_confusion_matrices = confusion_matrices.get("dummy", [])
basic_confusion_matrices = confusion_matrices.get("basic", [])
wavelets_confusion_matrices = confusion_matrices.get("wavelets", [])
auto_regr_confusion_matrices = confusion_matrices.get("auto_regr", [])
ngram_graphs_confusion_matrices = confusion_matrices.get("n_gram_graphs", [])


"""Check normality assumptions and if data are normally distributed, use ANOVA"""

print("Dummy Classifier Accuracies Normality (SW):", shapiro(dummy_accuracies))
print("Basic Classifier Accuracies Normality (SW):", shapiro(basic_accuracies))
print("Wavelet Accuracies Normality (SW):", shapiro(wavelets_accuracies))
print("AR Accuracies Normality (SW):", shapiro(auto_regr_accuracies))
print("N-Gram-Graphs Accuracies Normality (SW):", shapiro(ngram_graphs_accuracies))

print("Dummy Classifier F1 Macro Normality (SW):", shapiro(dummy_f1_macro))
print("Basic Classifier F1 Macro Normality (SW):", shapiro(basic_f1_macro))
print("Wavelet F1 Macro Normality (SW):", shapiro(wavelets_f1_macro))
print("AR F1 Macro Normality (SW):", shapiro(auto_regr_f1_macro))
print("N-Gram-Graphs F1 Macro Normality (SW):", shapiro(ngram_graphs_f1_macro))

print("Dummy Classifier Accuracies KS Test:", kstest(dummy_accuracies, 'norm'))
print("Basic Classifier Accuracies KS Test:", kstest(basic_accuracies, 'norm'))
print("Wavelet Accuracies KS Test:", kstest(wavelets_accuracies, 'norm'))
print("AR Accuracies KS Test:", kstest(auto_regr_accuracies, 'norm'))
print("N-Gram-Graphs Accuracies KS Test:", kstest(ngram_graphs_accuracies, 'norm'))

print("Dummy Classifier F1 Macro KS Test:", kstest(dummy_f1_macro, 'norm'))
print("Basic Classifier F1 Macro KS Test:", kstest(basic_f1_macro, 'norm'))
print("Wavelet F1 Macro KS Test:", kstest(wavelets_f1_macro, 'norm'))
print("AR F1 Macro KS Test:", kstest(auto_regr_f1_macro, 'norm'))
print("N-Gram-Graphs F1 Macro KS Test:", kstest(ngram_graphs_f1_macro, 'norm'))

print("Dummy Classifier Accuracies AD Test:", anderson(dummy_accuracies))
print("Basic Classifier Accuracies AD Test:", anderson(basic_accuracies))
print("Wavelet Accuracies AD Test:", anderson(wavelets_accuracies))
print("AR Accuracies AD Test:", anderson(auto_regr_accuracies))
print("N-Gram-Graphs Accuracies AD Test:", anderson(ngram_graphs_accuracies))

print("Dummy Classifier F1 Macro AD Test:", anderson(dummy_f1_macro))
print("Basic Classifier F1 Macro AD Test:", anderson(basic_f1_macro))
print("Wavelet F1 Macro AD Test:", anderson(wavelets_f1_macro))
print("AR F1 Macro AD Test:", anderson(auto_regr_f1_macro))
print("N-Gram-Graphs Accuracies AD Test:", anderson(ngram_graphs_f1_macro))

output_dir = RESULTS_TXT_PATH
os.makedirs(output_dir, exist_ok=True)

# Placeholder for normality test results
normality_results_path = os.path.join(output_dir, "normality_tests_results.txt")

# Perform and save normality test results
with open(normality_results_path, "w") as f:
    # Shapiro-Wilk Test for Accuracies
    f.write("Shapiro-Wilk Test for Accuracies:\n")
    f.write(f"Dummy Classifier: {shapiro(dummy_accuracies)}\n")
    f.write(f"Basic Classifier: {shapiro(basic_accuracies)}\n")
    f.write(f"Wavelet: {shapiro(wavelets_accuracies)}\n")
    f.write(f"Auto-Regr: {shapiro(auto_regr_accuracies)}\n")
    f.write(f"N-Gram-Graphs: {shapiro(ngram_graphs_accuracies)}\n\n")

    # Shapiro-Wilk Test for F1 Macro
    f.write("Shapiro-Wilk Test for F1 Macro:\n")
    f.write(f"Dummy Classifier: {shapiro(dummy_f1_macro)}\n")
    f.write(f"Basic Classifier: {shapiro(basic_f1_macro)}\n")
    f.write(f"Wavelet: {shapiro(wavelets_f1_macro)}\n")
    f.write(f"Auto-Regr: {shapiro(auto_regr_f1_macro)}\n")
    f.write(f"N-Gram-Graphs: {shapiro(ngram_graphs_f1_macro)}\n\n")

    # Kolmogorov-Smirnov Test for Accuracies
    f.write("Kolmogorov-Smirnov Test for Accuracies:\n")
    f.write(f"Dummy Classifier: {kstest(dummy_accuracies, 'norm')}\n")
    f.write(f"Basic Classifier: {kstest(basic_accuracies, 'norm')}\n")
    f.write(f"Wavelet: {kstest(wavelets_accuracies, 'norm')}\n")
    f.write(f"Auto-Regr: {kstest(auto_regr_accuracies, 'norm')}\n")
    f.write(f"N-Gram-Graphs: {kstest(ngram_graphs_accuracies, 'norm')}\n\n")

    # Kolmogorov-Smirnov Test for F1 Macro
    f.write("Kolmogorov-Smirnov Test for F1 Macro:\n")
    f.write(f"Dummy Classifier: {kstest(dummy_f1_macro, 'norm')}\n")
    f.write(f"Basic Classifier: {kstest(basic_f1_macro, 'norm')}\n")
    f.write(f"Wavelet: {kstest(wavelets_f1_macro, 'norm')}\n")
    f.write(f"Auto-Regr: {kstest(auto_regr_f1_macro, 'norm')}\n")
    f.write(f"N-Gram-Graphs: {kstest(ngram_graphs_f1_macro, 'norm')}\n\n")

    # Anderson-Darling Test for Accuracies
    f.write("Anderson-Darling Test for Accuracies:\n")
    f.write(f"Dummy Classifier: {anderson(dummy_accuracies)}\n")
    f.write(f"Basic Classifier: {anderson(basic_accuracies)}\n")
    f.write(f"Wavelet: {anderson(wavelets_accuracies)}\n")
    f.write(f"Auto-Regr: {anderson(auto_regr_accuracies)}\n")
    f.write(f"N-Gram-Graphs: {anderson(ngram_graphs_accuracies)}\n\n")

    # Anderson-Darling Test for F1 Macro
    f.write("Anderson-Darling Test for F1 Macro:\n")
    f.write(f"Dummy Classifier: {anderson(dummy_f1_macro)}\n")
    f.write(f"Basic Classifier: {anderson(basic_f1_macro)}\n")
    f.write(f"Wavelet: {anderson(wavelets_f1_macro)}\n")
    f.write(f"Auto-Regr: {anderson(auto_regr_f1_macro)}\n")
    f.write(f"N-Gram-Graphs: {anderson(ngram_graphs_f1_macro)}\n")

print(f"Normality test results saved to {normality_results_path}")

"""QQ-Plots for Accuracies and F1-Macro scores"""
output_dir = QQ_PLOTS_PATH
os.makedirs(output_dir, exist_ok=True)

# Dummy Accuracies
stats.probplot(dummy_accuracies, dist="norm", plot=plt)
plt.title('Q-Q Plot for Dummy Accuracies')
plt.savefig(os.path.join(output_dir, "qq_dummy_accuracies.png"), dpi=300, bbox_inches="tight")
plt.close()

# Basic Accuracies
stats.probplot(basic_accuracies, dist="norm", plot=plt)
plt.title('Q-Q Plot for Basic Accuracies')
plt.savefig(os.path.join(output_dir, "qq_basic_accuracies.png"), dpi=300, bbox_inches="tight")
plt.close()

# Wavelets Accuracies
stats.probplot(wavelets_accuracies, dist="norm", plot=plt)
plt.title('Q-Q Plot for Wavelets Accuracies')
plt.savefig(os.path.join(output_dir, "qq_wavelets_accuracies.png"), dpi=300, bbox_inches="tight")
plt.close()

# Auto-regr Accuracies
stats.probplot(auto_regr_accuracies, dist="norm", plot=plt)
plt.title('Q-Q Plot for Auto-regr Accuracies')
plt.savefig(os.path.join(output_dir, "qq_auto_regr_accuracies.png"), dpi=300, bbox_inches="tight")
plt.close()

# NGGs Accuracies
stats.probplot(ngram_graphs_accuracies, dist="norm", plot=plt)
plt.title('Q-Q Plot for NGGs Accuracies')
plt.savefig(os.path.join(output_dir, "qq_nggs_accuracies.png"), dpi=300, bbox_inches="tight")
plt.close()

output_dir = QQ_PLOTS_PATH / "f1_macro"
# Dummy F1 Macro
stats.probplot(dummy_f1_macro, dist="norm", plot=plt)
plt.title('Q-Q Plot for Dummy F1 Macro')
plt.savefig(os.path.join(output_dir, "qq_dummy_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

# Basic F1 Macro
stats.probplot(basic_f1_macro, dist="norm", plot=plt)
plt.title('Q-Q Plot for Basic F1 Macro')
plt.savefig(os.path.join(output_dir, "qq_basic_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

# Wavelets F1 Macro
stats.probplot(wavelets_f1_macro, dist="norm", plot=plt)
plt.title('Q-Q Plot for Wavelets F1 Macro')
plt.savefig(os.path.join(output_dir, "qq_wavelets_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

# Auto-regr F1 Macro
stats.probplot(auto_regr_f1_macro, dist="norm", plot=plt)
plt.title('Q-Q Plot for Auto-regr F1 Macro')
plt.savefig(os.path.join(output_dir, "qq_auto_regr_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

# NGGs F1 Macro
stats.probplot(ngram_graphs_f1_macro, dist="norm", plot=plt)
plt.title('Q-Q Plot for NGGs F1 Macro')
plt.savefig(os.path.join(output_dir, "qq_nggs_f1_macro.png"), dpi=300, bbox_inches="tight")
plt.close()

"""
Statistical Tests
"""

all_accuracies = dummy_accuracies + basic_accuracies + \
    wavelets_accuracies + auto_regr_accuracies + ngram_graphs_accuracies
accuracy_labels = (
    ['Dummy'] * len(dummy_accuracies) +
    ['Basic'] * len(basic_accuracies) +
    ['Wavelets'] * len(wavelets_accuracies) +
    ['Auto-Regr'] * len(auto_regr_accuracies) +
    ['N-Gram Graphs'] * len(ngram_graphs_accuracies)
)

# Combine all F1-Macro scores into a single dataset
all_f1_macro = dummy_f1_macro + basic_f1_macro + \
    wavelets_f1_macro + auto_regr_f1_macro + ngram_graphs_f1_macro
f1_macro_labels = (
    ['Dummy'] * len(dummy_f1_macro) +
    ['Basic'] * len(basic_f1_macro) +
    ['Wavelets'] * len(wavelets_f1_macro) +
    ['Auto-Regr'] * len(auto_regr_f1_macro) +
    ['N-Gram Graphs'] * len(ngram_graphs_f1_macro)
)

# Non-Parametric Test: Friedman Test for Accuracies
friedman_acc = friedmanchisquare(dummy_accuracies, basic_accuracies,
                                 wavelets_accuracies, auto_regr_accuracies, ngram_graphs_accuracies)
print("Friedman Test for Accuracies:", friedman_acc)

# Define pairwise comparisons
pairs = [
    ("Dummy", "Basic"), ("Dummy", "Wavelets"), ("Dummy",
                                                "Auto-Regr"), ("Dummy", "N-Gram Graphs"),
    ("Basic", "Wavelets"), ("Basic", "Auto-Regr"), ("Basic", "N-Gram Graphs"),
    ("Wavelets", "Auto-Regr"), ("Wavelets", "N-Gram Graphs"),
    ("Auto-Regr", "N-Gram Graphs")
]

# Pairwise Wilcoxon rank-sum tests for Accuracies with Bonferroni correction
p_values_acc = []
for pair in pairs:
    group1 = [acc for acc, label in zip(
        all_accuracies, accuracy_labels) if label == pair[0]]
    group2 = [acc for acc, label in zip(
        all_accuracies, accuracy_labels) if label == pair[1]]
    _, p_value = ranksums(group1, group2)
    p_values_acc.append(p_value)

adjusted_p_acc = multipletests(p_values_acc, method='bonferroni')[1]

print("\nBonferroni-corrected Pairwise Wilcoxon rank-sum tests for Accuracies:")
for (pair, p_adj) in zip(pairs, adjusted_p_acc):
    print(f"{pair[0]} vs {pair[1]}: Adjusted p-value = {p_adj}")

# Non-Parametric Test: Friedman Test for F1-Macro Scores
friedman_f1 = friedmanchisquare(dummy_f1_macro, basic_f1_macro,
                                wavelets_f1_macro, auto_regr_f1_macro, ngram_graphs_f1_macro)
print("\nFriedman Test for F1-Macro Scores:", friedman_f1)

# Pairwise Wilcoxon rank-sum tests for F1-Macro Scores with Bonferroni correction
p_values_f1 = []
for pair in pairs:
    group1 = [f1 for f1, label in zip(
        all_f1_macro, f1_macro_labels) if label == pair[0]]
    group2 = [f1 for f1, label in zip(
        all_f1_macro, f1_macro_labels) if label == pair[1]]
    _, p_value = ranksums(group1, group2)
    p_values_f1.append(p_value)

adjusted_p_f1 = multipletests(p_values_f1, method='bonferroni')[1]

print("\nBonferroni-corrected Pairwise Wilcoxon rank-sum tests for F1-Macro Scores:")
for (pair, p_adj) in zip(pairs, adjusted_p_f1):
    print(f"{pair[0]} vs {pair[1]}: Adjusted p-value = {p_adj}")

# Save results
output_dir = RESULTS_DIR
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "statistical_analysis_results.txt")
with open(output_file, "w") as f:
    f.write("Friedman Test for Accuracies:\n")
    f.write(str(friedman_acc) + "\n\n")

    f.write("Bonferroni-corrected Pairwise Wilcoxon rank-sum tests for Accuracies:\n")
    for (pair, p_adj) in zip(pairs, adjusted_p_acc):
        f.write(f"{pair[0]} vs {pair[1]}: Adjusted p-value = {p_adj}\n")
    f.write("\n")

    f.write("Friedman Test for F1-Macro Scores:\n")
    f.write(str(friedman_f1) + "\n\n")

    f.write(
        "Bonferroni-corrected Pairwise Wilcoxon rank-sum tests for F1-Macro Scores:\n")
    for (pair, p_adj) in zip(pairs, adjusted_p_f1):
        f.write(f"{pair[0]} vs {pair[1]}: Adjusted p-value = {p_adj}\n")

# Save key results in CSV format
csv_output_file = os.path.join(output_dir, "statistical_results_summary.csv")
summary_data = {
    "Test": [
        "Friedman Test for Accuracies",
        "Friedman Test for F1-Macro Scores"
    ],
    "Result": [
        str(friedman_acc),
        str(friedman_f1)
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(csv_output_file, index=False)

print(
    f"Statistical analysis results saved to:\n- {output_file}\n- {csv_output_file}")

"""Create Boxplots of Results"""
model_names = ["Dummy", "Basic", "Wavelets", "Auto-Regr", "N-Gram Graphs"]
accuracies_data = [dummy_accuracies, basic_accuracies, wavelets_accuracies, auto_regr_accuracies, ngram_graphs_accuracies]
f1_macro_data = [dummy_f1_macro, basic_f1_macro, wavelets_f1_macro, auto_regr_f1_macro, ngram_graphs_f1_macro]

""" Barplot for Accuracy Scores"""

p_values_vs_dummy = {
    'Dummy': 1.0,
    'Basic': 1.0,
    'Wavelets': 0.376,
    'Auto-Regr': 0.0016,
    'N-Gram Graphs': 0.058
}

p_values_vs_auto = {
    'Wavelets': 0.0016,
    'N-Gram Graphs': 0.0018
}

# Colors
colors = ['lightblue' if name in ['Dummy', 'Basic'] else 'lightgreen' for name in model_names]

# Mean and SEM
means = [np.mean(d) for d in accuracies_data]
sems = [stats.sem(d) for d in accuracies_data]

# Significance formatter
def stars(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, means, yerr=sems, capsize=5, color=colors, edgecolor="black")
plt.axhline(0.351, color='gray', linestyle='--', linewidth=1.5, label='Chance Level (~35.1%)')

# Add stars for models significantly better than Dummy
for i, model in enumerate(model_names):
    p = p_values_vs_dummy.get(model, 1.0)
    s = stars(p)
    if s:
        plt.text(i, means[i] + 0.02, s, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Additional comparison: Auto-Regr vs others
for i, model in enumerate(model_names):
    if model in p_values_vs_auto:
        p = p_values_vs_auto[model]
        s = stars(p)
        if s:
            plt.text(i, means[i] + 0.05, s, ha='center', va='bottom', fontsize=14, color='blue', fontweight='bold')

# Final touches
plt.title("Classification Accuracy by Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.ylim(0, 0.7)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig(fr"{BAR_PLOTS_PATH}\barplot_accuracies_final.png", dpi=300)
plt.show()

"""Barplot for F1 Macro scores"""

# Class distribution for chance F1 estimation
class_probs = [210/599, 128/599, 139/599, 122/599]
chance_f1_macro = sum([p**2 for p in class_probs])  # ≈ 0.262

# Means and SEMs
means_f1 = [np.mean(data) for data in f1_macro_data]
sems_f1 = [stats.sem(data) for data in f1_macro_data]

p_values_vs_dummy = {
    'Dummy': 1.0,
    'Basic': 0.0016,
    'Wavelets': 0.0016,
    'Auto-Regr': 0.0016,
    'N-Gram Graphs': 0.0016
}

p_values_vs_auto = {
    'Basic': 0.0016,
    'Wavelets': 0.0016,
    'N-Gram Graphs': 0.0029
}

def stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, means_f1, yerr=sems_f1, capsize=5, color=colors, edgecolor="black")
plt.axhline(chance_f1_macro, color='gray', linestyle='--', linewidth=1.5, label='Chance Level (~26.2%)')

# Add significance stars vs Dummy
for i, model in enumerate(model_names):
    p = p_values_vs_dummy.get(model, 1.0)
    s = stars(p)
    if s:
        plt.text(i, means_f1[i] + 0.02, s, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add significance stars vs Auto-Regr (in blue)
for i, model in enumerate(model_names):
    if model in p_values_vs_auto:
        p = p_values_vs_auto[model]
        s = stars(p)
        if s:
            plt.text(i, means_f1[i] + 0.05, s, ha='center', va='bottom', fontsize=14, color='blue', fontweight='bold')

# Labels and formatting
plt.title("F1 Macro Score by Model", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("F1 Macro Score", fontsize=14)
plt.ylim(0, 0.8)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()

# Save
plt.savefig(fr"{BAR_PLOTS_PATH}\barplot_f1_macro_final.png", dpi=300)
plt.show()

"""Confusion Matrices"""

class_names = ["EA", "preSD", "postSD", "ictal"]

# Helper function to convert string matrices to integer numpy arrays
def parse_confusion_matrix_robust(matrix_str):
    try:
        sanitized_str = matrix_str.replace(" ", ",").replace("\n", "").replace(",,", ",")
        sanitized_str = sanitized_str.strip(",").replace("[,", "[").replace(",]", "]")
        return np.array(eval(sanitized_str))
    except Exception as e:
        print(f"Error parsing matrix: {matrix_str}\nException: {e}")
        return np.zeros((4, 4))

# Sum confusion matrices for each representation with enhanced parsing
summed_confusion_matrices = {}
for method, matrices in confusion_matrices.items():
    parsed_matrices = [parse_confusion_matrix_robust(matrix) for matrix in matrices]
    summed_confusion_matrices[method] = sum(parsed_matrices)
    
cm_ar = summed_confusion_matrices["auto_regr"]

# Merge preSD and postSD into "transitional"
cm_new = np.zeros((3, 3), dtype=int)

# New class order: control (0), transitional (1), ictal (2)
cm_new[0, 0] = cm_ar[0, 0]  # control -> control
cm_new[0, 1] = cm_ar[0, 1] + cm_ar[0, 2]  # control -> transitional
cm_new[0, 2] = cm_ar[0, 3]  # control -> ictal

cm_new[1, 0] = cm_ar[1, 0] + cm_ar[2, 0]  # transitional -> control
cm_new[1, 1] = cm_ar[1, 1] + cm_ar[2, 1] + cm_ar[1, 2] + cm_ar[2, 2]  # transitional -> transitional
cm_new[1, 2] = cm_ar[1, 3] + cm_ar[2, 3]  # transitional -> ictal

cm_new[2, 0] = cm_ar[3, 0]  # ictal -> control
cm_new[2, 1] = cm_ar[3, 1] + cm_ar[3, 2]  # ictal -> transitional
cm_new[2, 2] = cm_ar[3, 3]  # ictal -> ictal

# New class names
class_names_new = ["Endogenous", "Transitional", "Ictal"]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_new,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=class_names_new,
    yticklabels=class_names_new,
)
plt.title(f"Confusion Matrix for A-R Representation (3 classes)", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=14)
plt.ylabel("True Labels", fontsize=14)
plt.tight_layout()
save_path = r"D:\1.Thesis\3. Presentation\media\conf\confusion_matrix_ar_3.png"
plt.savefig(save_path, dpi=300)
plt.close()

# Plot and save each summed confusion matrix
output_dir = CONFUSION_MATRICES_PATH
os.makedirs(output_dir, exist_ok=True)

for method, matrix in summed_confusion_matrices.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Summed Confusion Matrix for {method.capitalize()} Representation", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.tight_layout()
    save_path = f"{output_dir}/confusion_matrix_{method}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
