from typing import Optional

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler


class FaissKNearestNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k

        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.index = faiss.IndexFlatL2(x.shape[1])
        self.index.add(x.astype(np.float32))

    def knn(self, x: np.ndarray):
        distances, indices = self.index.search(x.astype(np.float32), k=self.k)
        return distances, indices

    def predict(self, x: np.ndarray):
        distances, neighbors = self.knn(x)
        return self._predict(neighbors, self.y)

    @staticmethod
    def _predict(neighbors, targets):
        predictions = []
        for k_neighbors in neighbors:
            predictions.append(targets[k_neighbors].mean().item())

        return np.array(predictions)


class TabNetWrapper(TabNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.embedder(x)

        steps_output, M_loss = self.tabnet.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        return res


class SiameseSystem(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        cat_features_indices,
        cardinality,
        emb_size,
        lr=1e-3,
        weight_decay=1e-5,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = TabNetWrapper(
            input_dim=input_dim,
            output_dim=output_dim,
            cat_idxs=cat_features_indices,
            cat_dims=cardinality,
            cat_emb_dim=emb_size,
            n_d=8,
            n_a=8,
            n_steps=1,
            gamma=2,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=256,
            momentum=0.02,
            mask_type="sparsemax",
        )

        bilinear_dim = output_dim
        self.bilinear = nn.Bilinear(bilinear_dim, bilinear_dim, bilinear_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(bilinear_dim + 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bilinear_dim + 1, bilinear_dim + 1),
            nn.LayerNorm(bilinear_dim + 1),
            nn.SiLU(bilinear_dim + 1),
            nn.Dropout(dropout),
            nn.Linear(bilinear_dim + 1, 1),
            nn.ReLU()
            #             nn.SiLU(1)
        )

        self.loss = nn.MSELoss()

        self.save_hyperparameters(
            {"output_dim": output_dim, "lr": lr, "weight_decay": weight_decay, "dropout": dropout}
        )

    def forward(self, x_first, x_second, y_second):
        first_embeddings = self.encoder(x_first.float())
        second_embeddings = self.encoder(x_second.float())

        proj = self.bilinear(first_embeddings, second_embeddings)

        output = self.head(torch.cat([proj, torch.log(y_second).unsqueeze(1).float()], dim=1))

        return output

    def training_step(self, batch, batch_idx):
        x_first = batch[0]["x"].float()
        y_first = batch[0]["y"].float()

        x_second = batch[1]["x"].float()
        y_second = batch[1]["y"].float()

        output = self(x_first, x_second, y_second).squeeze(1)

        loss = self.loss(torch.log(y_first).float(), output.float())
        self.log("Train/loss", loss)

        prediction = torch.exp(torch.clip(output.detach(), 5, 15))

        return {"loss": loss, "target": y_first, "prediction": prediction}

    def training_epoch_end(self, outputs):
        targets = torch.cat([out["target"] for out in outputs], dim=0).flatten().cpu()
        predictions = torch.cat([out["prediction"] for out in outputs], dim=0).flatten().cpu().detach().cpu()

        assert len(targets.shape) == 1
        assert len(predictions.shape) == 1

        score = deviation_metric(targets.numpy(), predictions.numpy()).item()

        self.log("Train/Score", score)
        self.logger.experiment.add_histogram("Train/predictions", predictions, self.global_step)

    def validation_step(self, batch, batch_idx):
        x_first = batch[0]["x"].float()
        y_first = batch[0]["y"].float()

        outs = []

        for i in range(1, len(batch)):
            candidate = batch[i]

            x_second = candidate["x"].float()
            y_second = candidate["y"].float()

            outs.append(self(x_first, x_second, y_second))

        prediction = torch.exp(torch.clip(torch.cat(outs, dim=1).mean(dim=1), 5, 15))

        return {"target": y_first, "prediction": prediction}

    def validation_epoch_end(self, outputs):
        targets = torch.cat([out["target"] for out in outputs], dim=0).flatten().cpu()
        predictions = torch.cat([out["prediction"] for out in outputs], dim=0).flatten().cpu().detach().cpu()

        assert len(targets.shape) == 1
        assert len(predictions.shape) == 1

        score = deviation_metric(targets.numpy(), predictions.numpy()).item()
        self.log("Val/Score", score)
        self.logger.experiment.add_histogram("Val/predictions", predictions, self.global_step)

    def test_step(self, batch, batch_idx):
        x_first = batch[0]["x"].float()
        y_first = batch[0]["y"].float()

        outs = []

        for i in range(1, len(batch)):
            candidate = batch[i]

            x_second = candidate["x"].float()
            y_second = candidate["y"].float()

            outs.append(self(x_first, x_second, y_second))

        prediction = torch.exp(torch.clip(torch.cat(outs, dim=1).mean(dim=1), 5, 15))

        return {"target": y_first, "prediction": prediction}

    def test_epoch_end(self, outputs):
        targets = torch.cat([out["target"] for out in outputs], dim=0).flatten().cpu()
        predictions = torch.cat([out["prediction"] for out in outputs], dim=0).flatten().cpu().detach().cpu()

        assert len(targets.shape) == 1
        assert len(predictions.shape) == 1

        score = deviation_metric(targets.numpy(), predictions.numpy()).item()
        self.log("Test/Score", score)
        self.logger.experiment.add_histogram("Test/predictions", predictions, self.global_step)

    def predict_step(self, batch, batch_ids):
        x_first = batch[0]["x"].float()

        outs = []

        for i in range(1, len(batch)):
            candidate = batch[i]

            x_second = candidate["x"].float()
            y_second = candidate["y"].float()

            outs.append(self(x_first, x_second, y_second))

        prediction = torch.exp(torch.clip(torch.cat(outs, dim=1).mean(dim=1), 5, 15))

        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=12, mode="min")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "Val/Score", "frequency": 2}
