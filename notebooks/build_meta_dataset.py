from datasense.data_loader import load_all_datasets
from datasense.meta_features import MetaFeatureExtractor
from datasense.pipeline_evaluator import PipelineEvaluator
import pandas as pd

bundles = load_all_datasets()

extractor = MetaFeatureExtractor()
evaluator = PipelineEvaluator()

meta_dataset_rows = []

for bundle in bundles:
    print(f"Processing: {bundle.name}")

    # Ingredient 1: dataset fingerprint (Day 2-3)
    features = extractor.extract(bundle.dataframe, target_column=bundle.target_column)

    # Ingredient 2: which pipeline actually wins (Day 4)
    result = evaluator.evaluate(
        dataframe=bundle.dataframe,
        target_column=bundle.target_column,
        task_type=bundle.task_type,
        dataset_name=bundle.name,
    )

    # Combine both ingredients into one row
    features["dataset_name"] = bundle.name
    features["task_type"] = bundle.task_type
    features["best_pipeline"] = result.best_pipeline_name
    features["best_score"] = result.best_score

    meta_dataset_rows.append(features)

meta_dataset = pd.DataFrame(meta_dataset_rows)

# Reorder columns: identifying info and label first, meta-features after
priority_columns = ["dataset_name", "task_type", "best_pipeline", "best_score"]
other_columns = [c for c in meta_dataset.columns if c not in priority_columns]
meta_dataset = meta_dataset[priority_columns + other_columns]

print("\nFinal meta-dataset:")
print(meta_dataset)

output_path = "data/meta_dataset.csv"
meta_dataset.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}") 