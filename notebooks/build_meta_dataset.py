from datasense.data_loader import load_all_datasets
from datasense.meta_features import MetaFeatureExtractor
import pandas as pd

# Load our standardized dataset collection
bundles = load_all_datasets()

# One extractor instance, reused across all datasets - it holds no
# per-dataset state, so this is safe and avoids repeated object creation
extractor = MetaFeatureExtractor()

meta_dataset_rows = []

for bundle in bundles:
    print(f"Extracting meta-features for: {bundle.name}")

    features = extractor.extract(bundle.dataframe, target_column=bundle.target_column)

    # Attach identifying info so we know which row belongs to which dataset
    features["dataset_name"] = bundle.name
    features["task_type"] = bundle.task_type

    meta_dataset_rows.append(features)

# Assemble into a single DataFrame: one row per dataset
meta_dataset = pd.DataFrame(meta_dataset_rows)

# Reorder columns so identifying info comes first (readability)
identifying_columns = ["dataset_name", "task_type"]
other_columns = [c for c in meta_dataset.columns if c not in identifying_columns]
meta_dataset = meta_dataset[identifying_columns + other_columns]

print("\nMeta-dataset preview:")
print(meta_dataset)

# Persist to disk for use in later milestones (Day 5: training the meta-learner)
output_path = "data/meta_dataset.csv"
meta_dataset.to_csv(output_path, index=False)
print(f"\nSaved meta-dataset to {output_path}")