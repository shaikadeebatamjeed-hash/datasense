"""
data_loader.py

Provides a standardized way to load multiple benchmark datasets for
meta-feature extraction. Each dataset is returned as a DatasetBundle,
bundling together the dataframe, its target column name, and its task
type (classification or regression), so downstream code (meta-feature
extraction, pipeline evaluation) never needs to special-case individual
datasets.
"""
print("Script started")
from dataclasses import dataclass
import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
)


@dataclass
class DatasetBundle:
    """
    A single dataset, standardized for use across DataSense.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. "iris".
    dataframe : pd.DataFrame
        Full dataset, features and target combined in one table.
    target_column : str
        Name of the column in `dataframe` holding the label/target.
    task_type : str
        Either "classification" or "regression".
    """
    name: str
    dataframe: pd.DataFrame
    target_column: str
    task_type: str


def _load_sklearn_dataset(loader_function, name: str, task_type: str) -> DatasetBundle:
    """
    Shared helper: converts any scikit-learn toy dataset loader into our
    standardized DatasetBundle format.

    Every scikit-learn loader (load_iris, load_wine, etc.) follows the same
    interface when called with as_frame=True — it returns a Bunch object
    with a `.frame` attribute already containing features + target combined.
    This helper exploits that consistency so we don't repeat the same
    conversion logic five times.
    """
    raw_data = loader_function(as_frame=True)
    dataframe = raw_data.frame

    return DatasetBundle(
        name=name,
        dataframe=dataframe,
        target_column="target",  # sklearn's as_frame=True always names it this
        task_type=task_type,
    )


def load_all_datasets() -> list[DatasetBundle]:
    """
    Load our full initial collection of benchmark datasets.

    Returns
    -------
    
    list[DatasetBundle]
        One bundle per dataset, ready for meta-feature extraction.
    """

    

def load_all_datasets() -> list[DatasetBundle]:
    """
    Load our full initial collection of benchmark datasets.
    """

    dataset_specs = [
        (load_iris, "iris", "classification"),
        (load_wine, "wine", "classification"),
        (load_breast_cancer, "breast_cancer", "classification"),
        (load_diabetes, "diabetes", "regression"),
        (fetch_california_housing, "california_housing", "regression"),
    ]

    bundles = []

    for loader_function, name, task_type in dataset_specs:
        print(f"Loading {name}...")
        bundle = _load_sklearn_dataset(loader_function, name, task_type)
        print(f"Finished {name}")
        bundles.append(bundle)

    return bundles
    
    

if __name__ == "__main__":
    bundles = load_all_datasets()

    print(f"Loaded {len(bundles)} datasets\n")

    for bundle in bundles:
        print(bundle.name)