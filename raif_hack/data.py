import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer  # noqa
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler

from raif_hack.settings import CATEGORICAL_OHE_FEATURES, NUM_FEATURES, TARGET

logger = logging.getLogger(__name__)

# all given numeric features distributions should be mapped to log space

numeric_transformer = Pipeline(
    steps=[
        ("log_space", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
        ("imputer", SimpleImputer(strategy="median")),
        # ("imputer", KNNImputer()),
        ("scaler", StandardScaler()),
    ]
)

# all given categorical features have low cardinality and can be one-hot-encoded
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


def get_preprocessor(numeric_features=NUM_FEATURES, categorical_ohe_features=CATEGORICAL_OHE_FEATURES):
    preprocessor = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_transformer, numeric_features),
                        ("categorical", categorical_transformer, categorical_ohe_features),
                    ]
                ),
            )
        ]
    )

    return preprocessor


class RemoveScarceValues:
    def __init__(self, min_occurencies):
        self.min_occurencies = min_occurencies
        self.column_good_values = {}

    def fit(self, X, y):
        for column in X.columns:
            self.column_good_values[column] = X[column].value_counts() >= self.min_occurencies
            self.column_good_values[column] = self.column_good_values[column][
                self.column_good_values[column] == True
            ].index.values
        return self

    def transform(self, X):
        for column in X.columns:
            X.loc[~X[column].isin(self.column_good_values[column]), column] = np.nan

        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def get_tabnet_preprocessor(numeric_features, categorical_le_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_transformer, numeric_features),
                        ("categorical", "passthrough", categorical_le_features),
                    ],
                    n_jobs=-1,
                ),
            )
        ]
    )

    return preprocessor


def get_knn_preprocessor(numeric_non_log, numeric_log, knn_cat_features=CATEGORICAL_OHE_FEATURES):
    numeric_non_log_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_log_transformer = Pipeline(
        steps=[
            ("log_space", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # numeric_transformer = FeatureUnion(
    #     transformer_list=[
    #         ("numeric_non_log", numeric_non_log_transformer, numeric_non_log),
    #         ("numeric_log", numeric_log_transformer, numeric_log)
    #     ]
    # )

    # all given categorical features have low cardinality and can be one-hot-encoded
    categorical_transformer = Pipeline(
        steps=[
            ("remove_scarse", RemoveScarceValues(min_occurencies=5)),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric_log", numeric_log_transformer, numeric_log),
                        ("numeric_non_log", numeric_non_log_transformer, numeric_non_log),
                        ("categorical", categorical_transformer, knn_cat_features),
                    ]
                ),
            )
        ]
    )

    return preprocessor
