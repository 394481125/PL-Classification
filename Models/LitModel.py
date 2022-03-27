import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)
        # self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # model = LitModel()
        # trainer.validate()
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # model = LitModel()
        # trainer.test(model)
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        # model = LitModel()
        # trainer.predict(model)
        x, y = batch
        pred = self(x)
        return pred