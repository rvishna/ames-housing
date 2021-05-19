"""Script for training a regression model on the Ames Housing dataset."""
from operator import itemgetter

import mlflow
import numpy as np
import pandas as pd
from sklearn import (
    impute,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn_pandas import DataFrameMapper
from skopt import BayesSearchCV


def create_mappper(dataset: pd.DataFrame, label: str) -> DataFrameMapper:
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
    mapper = DataFrameMapper(
        [
            (
                [feature],
                [
                    impute.SimpleImputer(strategy="median"),
                    preprocessing.RobustScaler(),
                ],
            )
            for feature in numerical_features
        ]
        + [
            (
                [feature],
                [
                    impute.SimpleImputer(strategy="most_frequent"),
                    preprocessing.OrdinalEncoder(categories=categories),
                    preprocessing.RobustScaler(),
                ],
            )
            for feature, categories in ordinal_features
        ]
        + [
            (
                [feature],
                [
                    impute.SimpleImputer(strategy="most_frequent"),
                    preprocessing.OneHotEncoder(
                        drop="if_binary",
                        categories=[dataset[feature].unique().tolist()],
                    ),
                ],
            )
            for feature in categorical_features
        ]
    )
    return mapper


def main() -> None:
    """Entrypoint for training script."""

    np.random.seed(0)
    mlflow.sklearn.autolog()
    dataset = pd.read_csv("data/train.csv")
    label = "SalePrice"
    pipe = pipeline.Pipeline(
        [
            ("mapper", create_mappper(dataset, label)),
            ("elastic_net", linear_model.ElasticNet()),
        ]
    )
    params = {
        "elastic_net__alpha": (1e-4, 100.0, "log-uniform"),
        "elastic_net__l1_ratio": (0.0, 1.0, "uniform"),
    }
    bayes_search = BayesSearchCV(
        pipe, params, n_iter=5, n_jobs=-1, return_train_score=True
    )
    train_dataset, test_dataset = model_selection.train_test_split(
        dataset, test_size=0.2
    )
    bayes_search.fit(train_dataset, train_dataset[label])
    bayes_search.score(test_dataset, test_dataset[label])


if __name__ == "__main__":
    main()
