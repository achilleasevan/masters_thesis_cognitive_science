import os
from sklearn.model_selection import train_test_split
import json
from file_paths import *

TEST_SIZE = 0.1
output_name = TRAIN_TEST_SPLIT_JSON_PATH
data_dir = DATA_DIR
file_paths = []
labels = []

for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            if file.endswith(".mat"):
                file_paths.append(os.path.join(class_path, file))
                labels.append(class_dir)
                
train_files, test_files, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=TEST_SIZE, stratify=labels, random_state=42
)


# split_data = {
#     "train_files": train_files,
#     "test_files": test_files,
#     "train_labels": train_labels,
#     "test_labels": test_labels,
# }
# with open(output_name, "w") as f:
#     json.dump(split_data, f, indent=4)


# with open(output_name, "r") as f:
#     loaded_data = json.load(f)
    
# train_files = loaded_data["train_files"]
# test_files = loaded_data["test_files"]
# train_labels = loaded_data["train_labels"]
# test_labels = loaded_data["test_labels"]

# Take the training set and split it into a 90% training set and a 10% validation set, to train the graphs
graph_train_validation_split_output_name = TRAIN_VALIDATION_SPLIT_FOR_GRAPHS_JSON_PATH

graph_train_files, graph_validation_files, graph_train_labels, graph_validation_labels = train_test_split(
    train_files, train_labels, test_size=TEST_SIZE, stratify=train_labels, random_state=42
)

graph_train_validation_split = {
    "graph_train_files": graph_train_files,
    "graph_validation_files": graph_validation_files,
    "graph_train_labels": graph_train_labels,
    "graph_validation_labels": graph_validation_labels,
}

with open(graph_train_validation_split_output_name, "r") as f:
    loaded_data_graphs = json.load(f)
    
graph_train_files = loaded_data_graphs["graph_train_files"]
graph_validation_files = loaded_data_graphs["graph_validation_files"]
graph_train_labels = loaded_data_graphs["graph_train_labels"]
graph_validation_labels = loaded_data_graphs["graph_validation_labels"]