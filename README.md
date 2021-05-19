#  MLflow Applied to the Ames Housing Dataset

This repository is my attempt at getting as close as possible to productionizing ML using a Kaggle dataset as example. Many of the concepts used here can be extended to other ML projects. It uses the following tools:

- [scikit-learn](https://scikit-learn.org/) for creating the machine learning models.
- [sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) for creating a mapper object that transforms pandas dataframes to numpy arrays for use in sklearn pipelines.
- [mlflow](https://mlflow.org/) for managing the machine learning lifecycle.
- [scikit-optimize](https://scikit-optimize.github.io/stable/) for hyperparameter tuning using Bayesian optimization.

Some things not in scope, but potentially items for future exploration include:

- Data version control
- Parallelizing hyperparameter optimization
- Training on massive amounts of data
