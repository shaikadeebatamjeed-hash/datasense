# notebooks/test_meta_features.py  (temporary, exploratory — not committed to production code)
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

print("Project Root:", PROJECT_ROOT)
print("Source Path:", SRC_PATH)

sys.path.insert(0, str(SRC_PATH))

print("sys.path[0]:", sys.path[0])

from datasense.meta_features import MetaFeatureExtractor



from sklearn.datasets import load_iris
import pandas as pd

# Load a well-known toy dataset to sanity-check our extractor
iris = load_iris(as_frame=True)
df = iris.frame  # includes both features and the 'target' column

extractor = MetaFeatureExtractor()
result = extractor.extract(df, target_column="target")

for key, value in result.items():
    print(f"{key}: {value}") 