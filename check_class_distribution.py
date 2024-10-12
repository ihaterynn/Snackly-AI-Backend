import os
from collections import Counter

dataset_path = "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/train"
labels = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        label = os.path.basename(root)
        labels.append(label)

print(Counter(labels))

