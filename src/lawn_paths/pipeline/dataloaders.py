from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
            self,
            samples: List[Tuple[Path, Path]],
            transform,
            length: int = None,
    ) -> None:

        self.transform = transform
        self.samples = samples

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        idx = idx % len(self.samples)

        # read image and mask from memory
        image_path, mask_path = self.samples[idx]
        image = io.imread(image_path, plugin='tifffile').astype(np.float32)
        mask = np.load(mask_path).astype(np.int32)

        # apply normalization
        #         image_norm = self.transform(image=image)["image"]
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask)

        return image, torch.unsqueeze(mask, 0).int()
