# from pathlib import Path

import torch as th
from torch.utils.data import Dataset
from albumentations import BaseCompose


class Dataset(Dataset):
    def __init__(
        self,
        transforms: BaseCompose | None = None
    ) -> None:
        super().__init__()

        self.transforms = transforms
        ...


    def __len__(self) -> int:
        ...



    def __getitem__(self, idx) -> tuple[th.Tensor, th.Tensor]:
        ...


        image = ...
        mask = ...


        # Apply transformations
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']


        return image, mask # type: ignore
