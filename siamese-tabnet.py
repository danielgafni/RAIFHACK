import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_tabnet.tab_network import TabNet, TabNetNoEmbeddings
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from raif_hack.data import get_knn_preprocessor, get_tabnet_preprocessor
from raif_hack.metrics import deviation_metric
from raif_hack.models import FaissKNearestNeighbors
from raif_hack.settings import (
    CATEGORICAL_LE_FEATURES,
    CATEGORICAL_OHE_FEATURES,
    NUM_FEATURES,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from raif_hack.torch_utils import RaifDataModule

if __name__ == "__main__":
    dm = RaifDataModule(
        df_path=TRAIN_PATH,
        num_features=NUM_FEATURES,
        cat_features=CATEGORICAL_LE_FEATURES,
        knn_cat_features=[],
        prediction_path=TEST_PATH,
        train_size=0.6,
        k=20,
        num_workers=128,
        batch_size=512,
    )

    batch = next(iter(dm.train_dataloader()))

    cardinality = dm.cardinality
    emb_size = dm.emb_size
    cat_features_indices = dm.cat_features_indices

    input_dim = dm.train_dataset[0][0]["x"].shape[0]
    output_dim = 8

    dm.batch_size = 128

    system = SiameseSystem(
        input_dim=input_dim,
        output_dim=output_dim,
        cat_features_indices=cat_features_indices,
        cardinality=cardinality,
        emb_size=emb_size,
        lr=1e-2,
        weight_decay=1e-3,
        dropout=0.2,
    )

    with torch.no_grad():
        predictions = system(batch[0]["x"], batch[1]["x"], batch[1]["y"])

    logger = pl.loggers.TensorBoardLogger(
        "logs",
        name="TabNet",
        log_graph=False,
    )
    LOCAL_MODEL_CHECKPOINTS_DIR = "models"
    checkpoints = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(LOCAL_MODEL_CHECKPOINTS_DIR, str(logger.name), str(logger.version)),
        monitor="Val/Score",
        verbose=False,
        mode="min",
        save_top_k=1,
    )
    assert checkpoints.dirpath is not None

    log_lr = pl.callbacks.LearningRateMonitor(logging_interval="step")
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="Val/Score",
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode="min",
    )
    gpu = pl.callbacks.GPUStatsMonitor(temperature=True)
    callbacks = [
        checkpoints,
        early_stopping,
        log_lr,
        gpu,
    ]

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        auto_lr_find=True,
        callbacks=callbacks,
        accumulate_grad_batches=16,
        num_sanity_val_steps=0,
        stochastic_weight_avg=True,
        max_epochs=30,
        log_every_n_steps=10,
    )

    trainer.fit(system, datamodule=dm)

    test_results = trainer.test(system, datamodule=dm, ckpt_path="best")
    predictions = trainer.predict(system, datamodule=dm)

    dm.predictions[TARGET] = predictions.flatten().cpu().numpy()

    dm.predictions[["id", TARGET]].to_csv(str(SUBMISSION_PATH), index=False)
