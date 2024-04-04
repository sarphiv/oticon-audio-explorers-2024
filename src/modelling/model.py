from typing import Literal

import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_optimizer import create_optimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import kornia as K

from modelling.metrics import binary_stat_scores, intersection_over_union, dice, recall, precision, accuracy



class Model(pl.LightningModule):
    def __init__(
            self,
            threshold: float,
            n_channels: int,
            n_classes: int,
            learning_rate_max: float,
            learning_rate_min: float,
            learning_rate_half_period: int,
            learning_rate_mult_period: int,
            learning_rate_warmup_max: float,
            learning_rate_warmup_steps: int,
            weight_decay: float
        ):
        super().__init__()
        self.save_hyperparameters()

        self.threshold = threshold
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = ...

        self.learning_rate_max = learning_rate_max
        self.learning_rate_min = learning_rate_min
        self.learning_rate_half_period = learning_rate_half_period
        self.learning_rate_mult_period = learning_rate_mult_period
        self.learning_rate_warmup_max = learning_rate_warmup_max
        self.learning_rate_warmup_period = learning_rate_warmup_steps
        self.weight_decay = weight_decay



    def forward(self, image: th.Tensor) -> th.Tensor:
        # Run inference
        logits = self.model(image)

        return logits


    def _shared_step(self, batch: th.Tensor, batch_idx: int, stage: Literal["train", "val", "test"]) -> th.Tensor:
        # Deconstruct batch
        images, mask = batch


        # Forward pass of model into logits
        logits = self.forward(images)

        # Calculation of loss
        loss = K.losses.tversky_loss(logits, mask, alpha=0.5, beta=0.5, eps=1e-6)


        # Calculate metrics
        tp, fp, tn, fn = binary_stat_scores(F.sigmoid(logits.detach()) > self.threshold, mask)

        # NOTE: IoU is the same as the Jaccard index
        iou_metric = intersection_over_union(tp, fp, fn).nanmean()
        # NOTE: Dice coefficient is the same as the F1 score
        dice_metric = dice(tp, fp, fn).nanmean()

        recall_metric = recall(tp, fn).nanmean()
        precision_metric = precision(tp, fp).nanmean()
        accuracy_metric = accuracy(tp, fp, tn, fn).nanmean()

        # Log metrics
        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_iou": iou_metric,
                f"{stage}_dice": dice_metric,
                f"{stage}_recall": recall_metric,
                f"{stage}_precision": precision_metric,
                f"{stage}_accuracy": accuracy_metric,
            },
            prog_bar=True,
        )


        # Return for backwards
        return loss



    def training_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        return self._shared_step(batch, batch_idx, "train")


    def validation_step(self, batch: th.Tensor, batch_idx: int) -> th.Tensor:
        return self._shared_step(batch, batch_idx, "val")



    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = create_optimizer(
            self.model, # type: ignore
            "adan",
            lr=self.learning_rate_max,
            weight_decay=self.weight_decay,
            use_lookahead=True,
            use_gc=True,
            eps=1e-6
        )

        # NOTE: Must instantiate cosine scheduler first,
        #  because super scheduler mutates the initial learning rate.
        lr_cosine = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.learning_rate_half_period,
            T_mult=self.learning_rate_mult_period,
            eta_min=self.learning_rate_min
        )
        lr_super = th.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.learning_rate_warmup_max,
            total_steps=self.learning_rate_warmup_period,
        )
        lr_scheduler = th.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[lr_super, lr_cosine], # type: ignore
            milestones=[self.learning_rate_warmup_period],
        )

        return { # type: ignore
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }
