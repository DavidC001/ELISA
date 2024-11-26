import glob
import os

from torch.utils.data import Dataset

from configuration import DatasetConfig


class MultiMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        mask_dir: os.PathLike,
        *args,
        **kwargs,
    ):
        self.images_folder = image_dir
        self.masks_folder = mask_dir

        self.images = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.masks = glob.glob(os.path.join(mask_dir, "*.jpg"))

        assert len(self.images) == len(self.masks)

    @classmethod
    def from_config(cls, config: DatasetConfig):
        return cls(**config.__dict__)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_folder + "/" + os.path.basename(img_path).replace(".jpg", ".jpg")

        return img_path, mask_path
