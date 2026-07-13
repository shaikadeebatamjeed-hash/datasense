"""
pipeline_evaluator.py

Evaluates multiple candidate ML pipelines on a given dataset and identifies
which one performs best. This produces the "label" half of our meta-dataset
(the meta-features from meta_features.py are the "input" half) - together
they form the training data for the meta-learner (Day 5).

A "pipeline" here refers to scikit-learn's Pipeline object: a chain of
preprocessing steps + a final model, treated as a single unit. This is a
deliberate echo of the project's own name - DataSense recommends ML
pipelines, so it's fitting that we evaluate real sklearn Pipeline objects,
not just bare models.
"""

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


@dataclass
class EvaluationResult:
    """
    Result of evaluating all candidate pipelines on one dataset.

    Attributes
    ----------
    dataset_name : str
        Which dataset this result belongs to.
    scores : dict[str, float]
        Mapping of pipeline name -> mean cross-validation score.
    best_pipeline_name : str
        Name of the pipeline with the highest score.
    best_score : float
        The winning score itself.
    """
    dataset_name: str
    scores: dict
    best_pipeline_name: str
    best_score: float


class PipelineEvaluator:
    """
    Evaluates a fixed set of candidate ML pipelines using cross-validation,
    choosing the appropriate metric and model set based on task type.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _get_candidate_pipelines(self, task_type: str) -> dict:
        """
        Build the dictionary of candidate pipelines for a given task type.

        Every pipeline includes StandardScaler as a preprocessing step
        before the model. This matters because SVC/SVR and Logistic/Linear
        Regression are all sensitive to feature scale (e.g., a feature
        ranging 0-1000 can dominate one ranging 0-1 unless scaled), while
        Random Forest is scale-invariant but unaffected by scaling either
        way - so applying it uniformly is simple and harmless.
        """
        if task_type == "classification":
            return {
                "logistic_regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=self.random_state)),
                ]),
                "random_forest": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(random_state=self.random_state)),
                ]),
                "svm": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(random_state=self.random_state)),
                ]),
            }
        elif task_type == "regression":
            return {
                "linear_regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]),
                "random_forest": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=self.random_state)),
                ]),
                "svm": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVR()),
                ]),
            }
        else:
            raise ValueError(f"Unknown task_type: {task_type!r}")

    def _get_scoring_metric(self, task_type: str) -> str:
        """Return the appropriate cross_val_score scoring string per task type."""
        if task_type == "classification":
            return "f1_macro"
        elif task_type == "regression":
            return "r2"
        else:
            raise ValueError(f"Unknown task_type: {task_type!r}")

    def evaluate(self, dataframe, target_column: str, task_type: str, dataset_name: str) -> EvaluationResult:
        """
        Evaluate all candidate pipelines for this task type on the given
        dataset, using cross-validation, and identify the best performer.
        """
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        pipelines = self._get_candidate_pipelines(task_type)
        scoring_metric = self._get_scoring_metric(task_type)

        scores = {}
        for pipeline_name, pipeline in pipelines.items():
            fold_scores = cross_val_score(
                pipeline, X, y, cv=self.cv_folds, scoring=scoring_metric
            )
            scores[pipeline_name] = fold_scores.mean()

        best_pipeline_name = max(scores, key=scores.get)
        best_score = scores[best_pipeline_name]

        return EvaluationResult(
            dataset_name=dataset_name,
            scores=scores,
            best_pipeline_name=best_pipeline_name,
            best_score=best_score,
        ) 