# from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
# import albumentations as A

# from modelling.data.dataset import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        validation_split: float,
        *dataloader_args, **dataloader_kwargs
    ):
        super().__init__()

        if validation_split <= 0 or validation_split >= 1:
            raise ValueError(f"Validation split ({validation_split}) must be in ]0, 1[")

        self.validation_split = validation_split

        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs


    def setup(self, stage: str):
        # TODO: Add transforms
        # transforms = A.Compose([

        # ])

        self.dataset = ...

        self.dataset_train, self.dataset_val = random_split(
            self.dataset,
            [1-self.validation_split, self.validation_split]
        )


    def _disable_shuffle_arg(self, dataloader_args: tuple, dataloader_kwargs: dict) -> tuple[Any, Any]:
        dataloader_kwargs = {**dataloader_kwargs}

        if len(dataloader_args) > 2:
            dataloader_args = (*dataloader_args[:2], False, *dataloader_args[3:])

        if "shuffle" in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = False

        return dataloader_args, dataloader_kwargs


    def train_dataloader(self):
        return DataLoader(self.dataset_train, *self.dataloader_args, **self.dataloader_kwargs)

    def val_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)
        return DataLoader(self.dataset_val, *args, **kwargs)
