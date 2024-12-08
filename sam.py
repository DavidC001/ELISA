import glob
import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

import configuration as c


def show_anns(anns):
    """Copied from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb"""

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4)
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.7]])
        img[m] = color_mask
    ax.imshow(img)


class SAMDownloader:
    models = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    @staticmethod
    def download(model, dir):
        checkpoint_name = SAMDownloader.models[model].split("/")[-1]
        checkpoint_path = os.path.join(dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print("Downloading model...")
            import urllib.request

            urllib.request.urlretrieve(SAMDownloader.models[model], checkpoint_path)

        else:
            print("Model has already been downloaded.")

        return checkpoint_path


@dataclass
class SegmentationMask:
    pass


class SegmentationMaskExtractor:
    def __init__(self, sam_config: c.SAMConfig):
        self.config = sam_config

        checkpoint_path = SAMDownloader.download(self.config.model, self.config.checkpoint_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[self.config.model](checkpoint_path).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def __call__(self, path: os.PathLike | list[os.PathLike]):
        return self.extract(path)

    def extract(self, path: str | list[str]):
        """
        Extracts segmentation masks from an image or a list of images.
            :param path: Path to the image, list of image paths, or a directory containing images. Nested directories are not supported.
            :return: List of tuples where each tuple contains the path to an image and its corresponding segmentation masks.
        """

        if isinstance(path, list):
            res = [(img, self.segment_path(img)) for img in tqdm(path)]
        else:
            if os.path.isdir(path):
                res = [
                    (img, self.segment_path(os.path.join(path, img)))
                    for img in tqdm(os.listdir(path))
                ]

            else:
                res = [(str(path), self.segment_path(path))]

        return res

    def segment_path(self, image_path: os.PathLike) -> list[SegmentationMask]:
        """
        Extracts segmentation masks from an image.
            :param image_path: Path to the image.
            :return: List of segmentation masks.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.resize, self.config.resize))

        return self.segment_img(image)

    def segment_img(self, img: np.ndarray) -> list[SegmentationMask]:
        """
        Extracts segmentation masks from an image.
            :param img: Image as a numpy array.
            :return: List of segmentation masks.
        """
        try:
            masks = self.mask_generator.generate(img)

            masks.sort(key=(lambda x: x["area"]), reverse=True)
            if self.config.n_masks > 0:
                masks = masks[: self.config.n_masks]

            return masks

        except Exception as e:
            print(f"Error processing image: {e}")
            return []


if __name__ == "__main__":
    conf = c.load_yaml_config("config.yaml")
    extractor = SegmentationMaskExtractor(conf.sam)
    images = glob.glob(conf.dataset.image_dir + "/*.jpg")
    os.makedirs(conf.dataset.mask_dir, exist_ok=True)

    all_masks = extractor.extract(images)

    for img, img_masks in all_masks:
        out_folder = img.replace(conf.dataset.image_dir, conf.dataset.mask_dir).removesuffix(".jpg")
        os.makedirs(out_folder, exist_ok=True)

        for i, mask in enumerate(img_masks):
            out = os.path.join(out_folder, f"mask_{i}.png")
            cv2.imwrite(out, mask["segmentation"].astype(np.uint8) * 255)
