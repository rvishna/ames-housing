"""Script for training a regression model on the Ames Housing dataset."""
from functools import partial
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import hyperopt
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from hpsklearn.components import (
    ada_boost_regression,
    gradient_boosting_regression,
    random_forest_regression,
)
from hyperopt import hp
from sklearn import impute, model_selection, pipeline, preprocessing
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn_pandas import DataFrameMapper


def create_mappper(
    dataset: pd.DataFrame, label: str, use_one_hot_encoding: bool = False
) -> DataFrameMapper:
    """
    Returns a mapper object that transforms a pandas dataframe to numpy arrays
    suitable for use in sklearn estimators.
    """

    numerical_features = [
        column
        for column in dataset.columns
        if column != label and dataset.dtypes[column] != np.dtype("O")
    ]
    quality_categories = [["Po", "Fa", "TA", "Gd", "Ex"]]
    ordinal_features = [
        ("LotShape", [["Reg", "IR1", "IR2", "IR3"]]),
        ("LandSlope", [["Gtl", "Mod", "Sev"]]),
        ("ExterQual", quality_categories),
        ("ExterCond", quality_categories),
        ("BsmtQual", quality_categories),
        ("BsmtCond", quality_categories),
        ("BsmtExposure", [["No", "Mn", "Av", "Gd"]]),
        ("BsmtFinType2", [["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]]),
        ("HeatingQC", quality_categories),
        ("CentralAir", [["N", "Y"]]),
        ("KitchenQual", quality_categories),
        ("Functional", [["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]]),
        ("FireplaceQu", quality_categories),
        ("GarageFinish", [["Unf", "RFn", "Fin"]]),
        ("GarageQual", quality_categories),
        ("GarageCond", quality_categories),
        ("PavedDrive", [["N", "P", "Y"]]),
        ("PoolQC", quality_categories),
    ]
    categorical_features = set(dataset.columns) - (
        {label}
        .union(numerical_features)
        .union(set(map(itemgetter(0), ordinal_features)))
    )

    features = []
    for feature in numerical_features:
        features.append(
            (
                [feature],
                [
                    impute.SimpleImputer(strategy="median"),
                    preprocessing.RobustScaler(),
                ],
            )
        )
    for feature, categories in ordinal_features:
        features.append(
            (
                [feature],
                [
                    impute.SimpleImputer(strategy="most_frequent"),
                    preprocessing.OrdinalEncoder(categories=categories),
                    preprocessing.RobustScaler(),
                ],
            )
        )
    for feature in categorical_features:
        categories = [dataset[feature].unique().tolist()]
        if use_one_hot_encoding:
            features.append(
                (
                    [feature],
                    [
                        impute.SimpleImputer(strategy="most_frequent"),
                        preprocessing.OneHotEncoder(
                            categories=categories, drop="if_binary"
                        ),
                    ],
                )
            )
        else:
            features.append(
                (
                    [feature],
                    [
                        impute.SimpleImputer(strategy="most_frequent"),
                        preprocessing.OrdinalEncoder(categories=categories),
                        preprocessing.RobustScaler(),
                    ],
                )
            )
    mapper = DataFrameMapper(features)
    return mapper


def plot_feature_importances(
    estimator: BaseEstimator,
    feature_names: List[str],
    n_top: int = 10,
    filename: str = "feature_importances.png",
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Plot the importances of the `n_top` most important features."""

    _ = plt.figure(figsize=figsize)
    feature_importances = pd.DataFrame(
        estimator.feature_importances_, index=feature_names, columns=["importance"]
    )
    feature_importances.sort_values("importance", ascending=False, inplace=True)
    feature_importances = feature_importances.head(n_top)
    sns.barplot(x=feature_importances["importance"], y=feature_importances.index)
    plt.title("Random forest feature importance")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature names")
    plt.savefig(filename)


def objective(  # pylint:disable=invalid-name
    regressor: BaseEstimator,
    X: Any,
    y: Any,
    run_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Computes the objective function for `hyperopt`."""

    run_name = (
        "Iteration "
        + str(run_params["iteration"])
        + "/"
        + str(run_params["max_iterations"])
    )
    run_params["iteration"] += 1
    with mlflow.start_run(nested=True, run_name=run_name):
        with ignore_warnings(category=ConvergenceWarning):
            scores = model_selection.cross_val_score(regressor, X, y, scoring="r2")
    return {
        "loss": np.mean(-scores),
        "loss_variance": np.var(-scores, ddof=1),
        "status": hyperopt.STATUS_OK,
    }


def log_best_model(
    model: BaseEstimator,
    mapper: DataFrameMapper,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    label: str,
) -> None:
    """Evaluates and logs the best model."""

    pipe = pipeline.Pipeline([("mapper", mapper), ("regressor", model)])
    with ignore_warnings(category=ConvergenceWarning):
        pipe.fit(train_dataset, train_dataset[label])
        if hasattr(model, "feature_importances_"):
            plot_feature_importances(
                model,
                mapper.transformed_names_,
                filename="feature_importances.png",
            )
            mlflow.log_artifact("feature_importances.png")
        mlflow.sklearn.log_model(pipe, "model")
        metrics = mlflow.sklearn.eval_and_log_metrics(
            pipe, test_dataset, test_dataset[label], prefix="test_"
        )
        print(metrics["test_r2_score"])


def log_data_artifacts(
    train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, label: str
) -> None:
    """Logs metadata about the dataset in the parent run."""

    mlflow.log_metrics(
        {
            "training_size": len(train_dataset),
            "test_size": len(test_dataset),
            "num_features": len(train_dataset.columns),
        }
    )
    mlflow.log_params({"label": label})


def main() -> None:
    """Entrypoint for training script."""

    np.random.seed(0)

    mlflow.set_experiment("Automated house prices regression")
    mlflow.sklearn.autolog(log_models=False, silent=True)

    dataset: pd.DataFrame = pd.read_csv("data/train.csv")
    dataset.drop(columns=["Id"], inplace=True)
    train_dataset, test_dataset = model_selection.train_test_split(
        dataset, test_size=0.2
    )

    label = "SalePrice"
    mapper = create_mappper(dataset, label)
    X_train = mapper.fit_transform(train_dataset)  # pylint:disable=invalid-name
    y_train = train_dataset[label]

    search_space = hp.choice(
        "regressor",
        [
            random_forest_regression("random_forest"),
            ada_boost_regression("ada_boost"),
            gradient_boosting_regression("grad_boosting"),
        ],
    )

    with mlflow.start_run():
        run_params = {"iteration": 1, "max_iterations": 32}

        mlflow.log_metric("max_iterations", run_params["max_iterations"])
        log_data_artifacts(train_dataset, test_dataset, label)

        result = hyperopt.fmin(
            fn=partial(objective, X=X_train, y=y_train, run_params=run_params),
            space=search_space,
            algo=hyperopt.tpe.suggest,
            max_evals=run_params["max_iterations"],
        )

        model = hyperopt.space_eval(search_space, result)
        with mlflow.start_run(run_name="Best Model", nested=True):
            log_best_model(model, mapper, train_dataset, test_dataset, label)


if __name__ == "__main__":
    main()
