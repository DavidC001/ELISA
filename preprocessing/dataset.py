import glob
import os

from torch.utils.data import Dataset

from configuration import DatasetConfig, ProjectConfig


class MultiMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: os.PathLike,
        mask_dir: os.PathLike,
        fmt: str = "jpg",
        *args,
        **kwargs,
    ):
        self.images_folder = image_dir
        self.masks_folder = mask_dir
        self.fmt = fmt

        avl_imgs = glob.glob(os.path.join(image_dir, f"*.{self.fmt}"))
        avl_masks = [x for x in glob.glob(os.path.join(mask_dir, "*")) if os.path.isdir(x)]

        # enforce bijection between images and masks
        img_names = set(os.path.basename(x).removesuffix(f".{fmt}") for x in avl_imgs)
        mask_names = set(os.path.basename(x) for x in avl_masks)

        common_basenames = img_names & mask_names

        print(f"Found {len(img_names - common_basenames)} images without masks.")
        print(f"Found {len(mask_names - common_basenames)} masks without images.")
        print(f"Found {len(common_basenames)} common images and masks. Discarding the rest.")

        self.data_names = sorted([x for x in common_basenames])

    @classmethod
    def from_config(cls, config: DatasetConfig | ProjectConfig):
        """Loads the dataset from a configuration object."""
        if isinstance(config, ProjectConfig):
            config = config.dataset

        return cls(**config.__dict__)

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        name = self.data_names[idx]

        img_path = os.path.join(self.images_folder, f"{name}.{self.fmt}")
        masks = glob.glob(os.path.join(self.masks_folder, name, "*.png"))

        assert len(masks) > 0, f"No masks found for image {img_path}."
        assert os.path.exists(img_path), f"Image {img_path} does not exist."

        return img_path, masks
