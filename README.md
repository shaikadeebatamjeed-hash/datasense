# DataSense: Learning Dataset Characteristics for Intelligent ML Pipeline Recommendation

DataSense is a meta-learning research project that predicts which machine learning
pipeline is likely to perform best on a new, unseen dataset — based on that dataset's
characteristics (number of rows, feature types, class balance, correlation structure, etc.).

## Status
🚧 Work in progress — Version 1 under active development.

## Project Structure
- `src/datasense/` — core Python package (meta-feature extraction, pipeline evaluation, meta-learner)
- `tests/` — unit tests
- `notebooks/` — exploratory experiments
- `data/` — datasets (not tracked in Git; see setup instructions below)

## Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.venv\Scripts\Activate.ps1` (PowerShell)
3. Install dependencies: `pip install -r requirements.txt`

## License
TBD
##Author 
Shaik Adeeba Tamjeed 