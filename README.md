# Missing Value Imputation Recommendation System Skeleton

## Retained imputers

The current candidate imputation pool is intentionally narrowed to the methods below:

1. `deletion`
2. `diffputer`
3. `grape`
4. `mean`
5. `median`
6. `knni`
7. `em`
8. `mice`
9. `missforest`
10. `gain`
11. `miwae`
12. `hivae`
13. `iterative_xgboost`
14. `nomi`

## Notes

- `deletion` is a listwise-deletion baseline and removes rows containing missing values.
- `grape` is a GRAPE-inspired pure-PyTorch approximation of bipartite graph edge prediction for imputation.
- `knni` is the single retained KNN imputation entry in the registry.
- `em` is implemented as a Gaussian EM imputer with mean fallback for numerical instability.
- `mice` is implemented as a multiple-imputation wrapper around sklearn `IterativeImputer`.
- `missforest` is implemented as a practical approximation using sklearn `IterativeImputer` with `RandomForestRegressor`.
- `diffputer`, `grape`, `gain`, and `miwae` require `torch`.
- `hivae` requires `tensorflow`, `tensorflow-probability`, and `tf-keras` in the current environment.
- `iterative_xgboost` requires `xgboost` and is exposed as an optional nonlinear iterative imputer.
- `nomi` requires `tensorflow`, `hnswlib`, and `neural_tangents`.


## Built-in Datasets

The project now ships with a small built-in dataset collection under `data/raw/`.

- `iris`: real numeric dataset, load with `label_column="species"` and `drop_columns=["target"]`
- `wine`: real numeric dataset, load with `label_column="target_name"` and `drop_columns=["target"]`
- `seeds`: numeric-only local research matrix with 7 features
- `pendigits`: real numeric dataset with 16 features, load with `label_column="target"`
- `twomoons`: deterministic synthetic nonlinear clustering benchmark
- `flame`: deterministic project-local flame-like clustering benchmark

You can inspect dataset metadata through `data/raw/dataset_manifest.json` or call `datasets.list_available_datasets(...)`.
