import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from alphaclip import AlphaCLIPEncoder
from configuration import ProjectConfig, load_yaml_config
from sam import SegmentationMaskExtractor


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    return intersection / union


class PreprocessPipeline:
    def __init__(self, config: ProjectConfig):
        self.sme = SegmentationMaskExtractor(config.sam)
        self.ace = AlphaCLIPEncoder(config.alphaclip)
        self.dataset = config.dataset

    def run_all(self):
        res = []
        for image in tqdm(glob.glob(self.dataset.image_dir + "/*.jpg")):
            try:
                res.append(self.run(image))
            except Exception as e:
                print(f"Error in {image}: {e}")

        return res

    def run(self, img_path: str):
        with torch.no_grad():
            gt_masks = self.create_gt_masks(img_path)
            sam_masks = self.create_sam_masks(img_path)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.reshape_image(img)
            img = Image.fromarray(img)

            gt_masks = [self.reshape_image(mask) for mask in gt_masks]
            sam_masks = [self.reshape_image(mask) for mask in sam_masks]

            sam_masks = self.remove_gt_masks(sam_masks, gt_masks)

            sam_shapes = self.get_shapes_from_masks(sam_masks)
            gt_shapes = self.get_shapes_from_masks(gt_masks)

            gt_embs = [self.ace.get_visual_embedding(img, mask) for mask in gt_masks]
            sam_embs = [self.ace.get_visual_embedding(img, mask) for mask in sam_masks]

            return {
                "img": os.path.basename(img_path),
                "gt_embs": [emb.cpu().numpy().tolist() for emb in gt_embs],
                "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
                "gt_shapes": gt_shapes,
                "sam_shapes": sam_shapes,
            }

    def create_gt_masks(self, img_path: str) -> list[np.ndarray]:
        size = Image.open(img_path).size

        with open(img_path.replace(".jpg", ".json")) as f:
            data = json.load(f)

            gt_shapes = [s for s in data["shapes"] if s["label"] == "target"]
            gt_masks = []

            for shape in gt_shapes:
                img = Image.new("L", size, 0)
                draw = ImageDraw.Draw(img)
                points = [tuple(point) for point in shape["points"]]

                draw.polygon(points, fill=255)

                mask = np.array(img)
                gt_masks.append(mask)

        return gt_masks

    def remove_gt_masks(self, sam_masks, gt_masks):
        return [
            mask for mask in sam_masks if not any(iou(mask, gt_mask) > 0.95 for gt_mask in gt_masks)
        ]

    def create_sam_masks(self, img_path: str) -> list[np.ndarray]:
        res = self.sme.segment_path(img_path)
        sam_masks = [mask["segmentation"].astype("uint8") * 255 for mask in res]

        return sam_masks

    def reshape_image(self, image: np.ndarray, size: int = 1024):
        return cv2.resize(image, (size, size))

    def get_shapes_from_masks(self, masks: list[np.ndarray]):
        shapes = [
            cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for mask in masks
        ]

        return [[[tuple(p[0]) for p in s] for s in shape] for shape in shapes]

    def inference_preprocess(self, img_path: str):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.reshape_image(img)
        img = Image.fromarray(img)

        sam_masks = self.create_sam_masks(img_path)
        sam_embs = [self.ace.get_visual_embedding(img, mask) for mask in sam_masks]
        sam_shapes = self.get_shapes_from_masks(sam_masks)

        return {
            "img": os.path.basename(img_path),
            "sam_shapes": sam_shapes,
            "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
        }


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    pipeline = PreprocessPipeline(config)

    res = pipeline.run_all()

    with open(f"data/{os.path.basename(config.dataset.image_dir)}-v2.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
