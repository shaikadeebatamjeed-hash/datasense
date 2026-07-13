from datasense.data_loader import load_all_datasets
from datasense.pipeline_evaluator import PipelineEvaluator

bundles = load_all_datasets()
evaluator = PipelineEvaluator()

for bundle in bundles:
    print(f"Evaluating: {bundle.name} ({bundle.task_type})")
    result = evaluator.evaluate(
        dataframe=bundle.dataframe,
        target_column=bundle.target_column,
        task_type=bundle.task_type,
        dataset_name=bundle.name,
    )
    print(f"  Scores: {result.scores}")
    print(f"  Best pipeline: {result.best_pipeline_name} (score={result.best_score:.4f})")
    print()