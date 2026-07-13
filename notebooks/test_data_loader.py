from datasense.data_loader import load_all_datasets

print("Import successful")


# Load our full initial collection of datasets
bundles = load_all_datasets()

print(f"Loaded {len(bundles)} datasets.\n")

for bundle in bundles:
    print(f"Name: {bundle.name}")
    print(f"  Task type: {bundle.task_type}")
    print(f"  Shape: {bundle.dataframe.shape}")
    print(f"  Target column: {bundle.target_column}")
    print()

