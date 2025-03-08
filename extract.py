import os

# Path to the 'interf' folder
interf_folder = os.path.join("dataset", "deformationDetection", "interf")

# List all files in the 'interf' folder
sample_names = [f for f in os.listdir(interf_folder) if os.path.isfile(os.path.join(interf_folder, f))]

# Remove file extensions (optional)
# sample_names = [os.path.splitext(f)[0] for f in sample_names]

# Split the dataset into train, val, and test sets
train_ratio = 0.7  # 70% for training
val_ratio = 0.2    # 20% for validation
test_ratio = 0.1   # 10% for testing

num_samples = len(sample_names)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
num_test = num_samples - num_train - num_val

train_samples = sample_names[:num_train]
val_samples = sample_names[num_train:num_train + num_val]
test_samples = sample_names[num_train + num_val:]

# Save the sample names to files
data_dir = os.path.join("dataset", "deformationDetection")

with open(os.path.join(data_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_samples))

with open(os.path.join(data_dir, "val.txt"), "w") as f:
    f.write("\n".join(val_samples))

with open(os.path.join(data_dir, "test.txt"), "w") as f:
    f.write("\n".join(test_samples))

print("Dataset split and saved to train.txt, val.txt, and test.txt")