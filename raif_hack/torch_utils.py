from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from raif_hack.data import get_knn_preprocessor, get_tabnet_preprocessor
from raif_hack.models import FaissKNearestNeighbors
from raif_hack.settings import (
    CATEGORICAL_LE_FEATURES,
    CATEGORICAL_OHE_FEATURES,
    NUM_FEATURES,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from raif_hack.data import RemoveScarceValues


class RaifKNNTrainDataset(Dataset):
    def __init__(
        self,
        x_train_0: np.ndarray,
        x_1: np.ndarray,
        y_train_0: np.ndarray,
        y_1: np.ndarray,
        x_1_knn: np.ndarray,
        model: FaissKNearestNeighbors,
    ):
        self.x_train_0 = x_train_0
        self.x_1 = x_1
        self.y_train_0 = y_train_0
        self.y_1 = y_1

        self.x_1_knn = x_1_knn

        self.model = model

        self.distances, self.indices = self.model.knn(self.x_1_knn)

    def __getitem__(self, item):
        neighbor_indices = self.indices[item]
        neighbor_index = np.random.choice(neighbor_indices, 1).item()

        x_first = self.x_1[item]
        y_first = self.y_1[item]

        x_second = self.x_train_0[neighbor_index]
        y_second = self.y_train_0[neighbor_index]

        samples = {0: {"x": x_first, "y": y_first}, 1: {"x": x_second, "y": y_second}}

        return samples

    def __len__(self):
        return len(self.x_1)


class RaifKNNValidationDataset(Dataset):
    def __init__(
        self,
        x_train_0: np.ndarray,
        x_1: np.ndarray,
        y_train_0: np.ndarray,
        y_1: np.ndarray,
        x_1_knn: np.ndarray,
        model: FaissKNearestNeighbors,
    ):
        self.x_train_0 = x_train_0
        self.x_1 = x_1
        self.y_train_0 = y_train_0
        self.y_1 = y_1

        self.x_1_knn = x_1_knn

        self.model = model

        self.distances, self.indices = self.model.knn(self.x_1_knn)

    def __getitem__(self, item):
        neighbor_indices = self.indices[item]

        x_first = self.x_1[item]
        y_first = self.y_1[item]

        samples = {0: {"x": x_first, "y": y_first}}

        for i, neighbor_index in enumerate(neighbor_indices):
            x_neighbor = self.x_train_0[neighbor_index]
            y_neighbor = self.y_train_0[neighbor_index]

            samples[i + 1] = {"x": x_neighbor, "y": y_neighbor}

        return samples

    def __len__(self):
        return len(self.x_1)


class RaifKNNPredictionDataset(Dataset):
    def __init__(
        self,
        x_train_0: np.ndarray,
        x_1: np.ndarray,
        y_train_0: np.ndarray,
        x_1_knn: np.ndarray,
        model: FaissKNearestNeighbors,
    ):
        self.x_train_0 = x_train_0
        self.x_1 = x_1
        self.y_train_0 = y_train_0

        self.x_1_knn = x_1_knn

        self.model = model

        self.distances, self.indices = self.model.knn(self.x_1_knn)

    def __getitem__(self, item):
        neighbor_indices = self.indices[item]

        x_first = self.x_1[item]

        samples = {
            0: {
                "x": x_first,
            }
        }

        for i, neighbor_index in enumerate(neighbor_indices):
            x_neighbor = self.x_train_0[neighbor_index]
            y_neighbor = self.y_train_0[neighbor_index]

            samples[i + 1] = {"x": x_neighbor, "y": y_neighbor}

        return samples

    def __len__(self):
        return len(self.x_1)


class HandleInvalidLabelEncoder:
    def __init__(self):
        self.classes = {}

    def fit(self, y):
        for value in y:
            if value in self.classes:
                pass
            else:
                self.classes[value] = len(self.classes) + 1

        return self

    def transform(self, y):
        transformed = []
        for value in y:
            try:
                transformed.append(self.classes[value])
            except KeyError:
                transformed.append(0)

        return np.array(transformed)


class RaifDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_path: str,
        num_features: List[str],
        cat_features: List[str],
        knn_cat_features: List[str],
        prediction_path: Optional[str] = None,
        train_size: float = 0.6,
        k: int = 2,
        batch_size: int = 128,
        num_workers: int = 8,
    ):
        super().__init__()

        self.df_path = df_path
        self.prediction_path = prediction_path
        self.num_features = num_features
        self.cat_features = cat_features
        self.knn_cat_features = knn_cat_features
        self.train_size = train_size
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_features_indices = list(range(len(num_features)))
        self.cat_features_indices = [i + len(num_features) for i in range(len(cat_features))]

        self.cardinality = [None for f in cat_features]
        self.emb_size = [3 for f in cat_features]

        self.df = None
        self.predictions = None
        self.train = None
        self.val = None
        self.test = None
        self.remove_scarse = None
        self.encoders = None
        self.tabnet_preprocessor = None
        self.knn_preprocessor = None
        self.knn_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predictions_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = pd.read_csv(self.df_path)
        df["date"] = pd.to_datetime(df["date"])
        self.df = df.copy()

        predictions = pd.read_csv(self.prediction_path)
        predictions["date"] = pd.to_datetime(predictions["date"])
        self.predictions = predictions.copy()

        self.train, self.val = train_test_split(df, test_size=1 - self.train_size, stratify=df["price_type"])
        self.val, self.test = train_test_split(self.val, test_size=0.5)

        self.train = pd.concat(
            [
                self.train,
                self.val[self.val["price_type"] == 0].reset_index(),
                self.test[self.test["price_type"] == 0].reset_index(),
            ]
        )

        self.val = self.val[self.val["price_type"] == 1].reset_index()
        self.test = self.test[self.test["price_type"] == 1].reset_index()

        train_0 = self.train[self.train["price_type"] == 0].reset_index().copy()
        train_1 = self.train[self.train["price_type"] == 1].reset_index().copy()

        val = self.val[self.num_features + self.cat_features].copy()
        test = self.test[self.num_features + self.cat_features].copy()
        predictions = predictions[self.num_features + self.cat_features].copy()

        y_train_0 = train_0[TARGET].values
        y_train_1 = train_1[TARGET].values
        y_val = self.val[TARGET].values
        y_test = self.test[TARGET].values

        train = self.train[self.num_features + self.cat_features].copy()
        train_0 = train_0[self.num_features + self.cat_features].copy()
        train_1 = train_1[self.num_features + self.cat_features].copy()

        self.remove_scarse = RemoveScarceValues(
            min_occurencies=50,
        ).fit(train[self.cat_features], None)

        train_0[self.cat_features] = self.remove_scarse.transform(train_0[self.cat_features])
        train_1[self.cat_features] = self.remove_scarse.transform(train_1[self.cat_features])
        val[self.cat_features] = self.remove_scarse.transform(val[self.cat_features])
        test[self.cat_features] = self.remove_scarse.transform(test[self.cat_features])
        predictions[self.cat_features] = self.remove_scarse.transform(predictions[self.cat_features])

        self.encoders = {}
        for cat_feature in self.cat_features:
            self.encoders[cat_feature] = HandleInvalidLabelEncoder().fit(train_0[cat_feature])

            train_0[cat_feature] = self.encoders[cat_feature].transform(train_0[cat_feature])
            train_1[cat_feature] = self.encoders[cat_feature].transform(train_1[cat_feature])
            val[cat_feature] = self.encoders[cat_feature].transform(val[cat_feature])
            test[cat_feature] = self.encoders[cat_feature].transform(test[cat_feature])
            predictions[cat_feature] = self.encoders[cat_feature].transform(predictions[cat_feature])

        self.tabnet_preprocessor = get_tabnet_preprocessor(
            numeric_features=self.num_features, categorical_le_features=self.cat_features
        ).fit(train_0)

        # num_features_to_log = (np.abs(
        #     train[self.num_features].mean() / train[self.num_features].median() - 1)).sort_values(ascending=False)[
        #                       :40].index.values.tolist()
        # num_features_non_log = [f for f in self.num_features if f not in num_features_to_log]
        self.knn_preprocessor = get_knn_preprocessor(
            numeric_non_log=["lng", "lat"], numeric_log=[], knn_cat_features=self.knn_cat_features
        ).fit(train_0)

        x_train_0 = np.ascontiguousarray(self.tabnet_preprocessor.transform(train_0))
        x_train_1 = np.ascontiguousarray(self.tabnet_preprocessor.transform(train_1))
        x_val = np.ascontiguousarray(self.tabnet_preprocessor.transform(val))
        x_test = np.ascontiguousarray(self.tabnet_preprocessor.transform(test))
        x_predictions = np.ascontiguousarray(self.tabnet_preprocessor.transform(predictions))

        for i, col in enumerate(self.cat_features):
            self.cardinality[i] = len(self.encoders[col].classes) + 1

        knn_x_train_0 = np.ascontiguousarray(self.knn_preprocessor.transform(train_0))
        knn_x_train_1 = np.ascontiguousarray(self.knn_preprocessor.transform(train_1))
        knn_x_val = np.ascontiguousarray(self.knn_preprocessor.transform(val))
        knn_x_test = np.ascontiguousarray(self.knn_preprocessor.transform(test))
        knn_x_predictions = np.ascontiguousarray(self.knn_preprocessor.transform(predictions))

        self.knn_model = FaissKNearestNeighbors(k=self.k)
        self.knn_model.fit(knn_x_train_0, None)

        self.train_dataset = RaifKNNTrainDataset(
            x_train_0=x_train_0,
            x_1=x_train_1,
            y_train_0=y_train_0,
            y_1=y_train_1,
            x_1_knn=knn_x_train_1,
            model=self.knn_model,
        )

        self.val_dataset = RaifKNNValidationDataset(
            x_train_0=x_train_0, x_1=x_val, y_train_0=y_train_0, y_1=y_val, x_1_knn=knn_x_val, model=self.knn_model
        )
        self.test_dataset = RaifKNNValidationDataset(
            x_train_0=x_train_0, x_1=x_test, y_train_0=y_train_0, y_1=y_test, x_1_knn=knn_x_test, model=self.knn_model
        )
        self.predictions_dataset = RaifKNNPredictionDataset(
            x_train_0=x_train_0, x_1=x_predictions, y_train_0=y_train_0, x_1_knn=knn_x_predictions, model=self.knn_model
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            #             sampler=self.train_sampler,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.predictions_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )
