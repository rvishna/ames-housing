"""Script for training a regression model on the Ames Housing dataset."""
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
    elasticnet,
    gradient_boosting_regression,
    random_forest_regression,
)
from hyperopt import hp
from sklearn import (
    impute,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
)
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


def main() -> None:
    """Entrypoint for training script."""

    np.random.seed(0)
    mlflow.sklearn.autolog()
    dataset: pd.DataFrame = pd.read_csv("data/train.csv")
    dataset.drop(columns=["Id"], inplace=True)
    train_dataset, test_dataset = model_selection.train_test_split(
        dataset, test_size=0.2
    )
    label = "SalePrice"

    def make_pipeline(regressor: BaseEstimator) -> pipeline.Pipeline:
        mapper = create_mappper(
            dataset,
            label,
            use_one_hot_encoding=isinstance(regressor, linear_model.ElasticNet),
        )
        return pipeline.Pipeline([("mapper", mapper), ("regressor", regressor)])

    def objective(regressor: BaseEstimator) -> Dict[str, Any]:
        pipe = make_pipeline(regressor)
        with ignore_warnings(category=ConvergenceWarning):
            scores = model_selection.cross_val_score(
                pipe, train_dataset, train_dataset[label], scoring="r2"
            )
        return {"loss": -scores.mean(), "status": hyperopt.STATUS_OK}

    search_space = hp.choice(
        "regressor",
        [
            elasticnet("elastic_net", max_iter=1000),
            random_forest_regression("random_forest"),
            ada_boost_regression("ada_boost"),
            gradient_boosting_regression("grad_boosting"),
        ],
    )
    result = hyperopt.fmin(
        fn=objective, space=search_space, algo=hyperopt.tpe.suggest, max_evals=10
    )
    pipe = make_pipeline(hyperopt.space_eval(search_space, result))
    with ignore_warnings(category=ConvergenceWarning):
        pipe.fit(train_dataset, train_dataset[label])
    r2_test = pipe.score(test_dataset, test_dataset[label])
    r2_train = pipe.score(train_dataset, train_dataset[label])
    print(r2_train, r2_test)
    if hasattr(pipe.steps[1][1], "feature_importances_"):
        plot_feature_importances(pipe.steps[1][1], pipe.steps[0][1].transformed_names_)


if __name__ == "__main__":
    main()
