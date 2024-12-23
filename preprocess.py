import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import math
from PIL import Image, ImageDraw
from tqdm import tqdm

from configuration import ProjectConfig, dataclass, load_yaml_config
from preprocessing.alphaclip import AlphaCLIPEncoder
from preprocessing.sam import SegmentationMaskExtractor

import fiftyone.zoo as foz
import fiftyone as fo


@dataclass
class InferenceSample:
    img: str
    sam_shapes: list[list[tuple]]
    sam_embs: list[np.ndarray]


@dataclass
class TrainingSample(InferenceSample):
    gt_shapes: list[list[tuple]]
    gt_embs: list[np.ndarray]


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


class PreprocessPipeline:
    def __init__(self, config: ProjectConfig):
        self.sme = SegmentationMaskExtractor(config.sam)
        self.ace = AlphaCLIPEncoder(config.alphaclip)
        self.COCO_dataset = config.dataset.COCO
        self.ReasonSeg_dataset = config.dataset.ReasonSeg

    def run_all(self, mask_only: bool = False, dataset_name: str = "ReasonSeg",) -> list[dict]:
        """
        Runs the pipeline on either the ReasonSeg dataset (images and JSONs in a folder)
        or the COCO dataset (loaded via FiftyOne).
        """
        res = []

        if dataset_name == "ReasonSeg":
            for image in tqdm(glob.glob(self.ReasonSeg_dataset.image_dir + "/*.jpg")):
                try:
                    res.append(self.run_ReasonSeg(image, mask_only))
                except Exception as e:
                    print(f"Error in {image}: {e}")

        else:
            # Load the COCO dataset via FiftyOne
            ds = foz.load_zoo_dataset(
                self.COCO_dataset.name,
                split="train",
                label_types=["segmentations"],
                max_samples=self.COCO_dataset.num_samples,
                shuffle=True,
                seed=42,
            )

            for sample in tqdm(ds):
                try:
                    res.append(self.run_COCO(sample, mask_only))
                except Exception as e:
                    print(f"Error in {sample.filepath}: {e}")

        return res

    def run_COCO(self, sample: fo.Sample, mask_only: bool) -> dict:
        """
        Run the preprocessing pipeline on a COCO sample (loaded via FiftyOne).

        Args:
            sample (fo.Sample): FiftyOne sample with segmentations.
            mask_only (bool): Whether to multiply the image by the mask or not.

        Returns:
            dict: Dictionary containing the same keys as ReasonSeg output:
                  {
                      "img": <image-file-name>,
                      "gt_embs": [...],
                      "sam_embs": [...],
                      "gt_shapes": [...],
                      "sam_shapes": [...]
                  }
        """
        # (1) Load the image in CV2 and convert
        img_path = sample.filepath
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # (2) Collect GT masks from the sample's segmentation field
        gt_masks = []
        
        image_width = sample.metadata.width
        image_height = sample.metadata.height
        
        if sample.ground_truth is not None:
            for detection in sample.ground_truth.detections:
                if detection.mask is not None:
                    # Extract bounding box coordinates
                    bbox = detection.bounding_box  # [top-left-x, top-left-y, width, height]
                    x1 = round(bbox[0] * image_width)
                    y1 = round(bbox[1] * image_height)
                    x2 = round((bbox[0] + bbox[2]) * image_width)
                    y2 = round((bbox[1] + bbox[3]) * image_height)
                    
                    mask = detection.mask 
                    # reshape the mask to the bounding box size if the shape does not match
                    if mask.shape[0] != y2 - y1 or mask.shape[1] != x2 - x1:
                        mask = torch.tensor(mask)
                        mask = T.Resize((y2 - y1, x2 - x1))(mask.unsqueeze(0).unsqueeze(0))
                        mask = mask.squeeze(0).squeeze(0).numpy()
                    
                    
                    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = mask * 255
                    
                    gt_masks.append(full_mask)

        # (3) Generate the SAM masks for this image
        sam_masks = self.create_sam_masks(img_path)

        # (4) Reshape images and masks to a unified size (e.g., 1024x1024)
        img = self.reshape_image(img)
        img_pil = Image.fromarray(img)

        gt_masks = [self.reshape_image(m) for m in gt_masks]
        sam_masks = [self.reshape_image(m) for m in sam_masks]
        

        # (5) Remove any SAM masks overlapping too much with GT
        sam_masks = self.remove_gt_masks(sam_masks, gt_masks)

        # (6) Extract shapes (contours) from the masks
        gt_shapes = self.get_shapes_from_masks(gt_masks)
        sam_shapes = self.get_shapes_from_masks(sam_masks)

        # (7) Compute embeddings
        gt_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in gt_masks]
        sam_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in sam_masks]

        # (8) Package results
        return {
            "img": os.path.basename(img_path),
            "gt_embs": [emb.cpu().numpy().tolist() for emb in gt_embs],
            "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
            "gt_shapes": gt_shapes,
            "sam_shapes": sam_shapes,
        }

    def run_ReasonSeg(self, img_path: str, mask_only: bool) -> dict:
        """
        Run the preprocessing pipeline on a ReasonSeg image.

        Args:
            img_path (str): Path to the image to preprocess.
            mask_only (bool): Whether to multiply the image by the mask or not.

        Returns:
            dict: Dictionary in the same format used by `run_COCO`.
        """
        with torch.no_grad():
            # create GT masks from the JSON annotation next to the image
            gt_masks = self.create_gt_masks(img_path)
            # create SAM masks
            sam_masks = self.create_sam_masks(img_path)

            # read and reshape the image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.reshape_image(img)
            img_pil = Image.fromarray(img)

            # resize masks
            gt_masks = [self.reshape_image(mask) for mask in gt_masks]
            sam_masks = [self.reshape_image(mask) for mask in sam_masks]

            # remove overlapping SAM masks
            sam_masks = self.remove_gt_masks(sam_masks, gt_masks)

            # shapes
            gt_shapes = self.get_shapes_from_masks(gt_masks)
            sam_shapes = self.get_shapes_from_masks(sam_masks)

            # embeddings
            gt_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in gt_masks]
            sam_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in sam_masks]

            return {
                "img": os.path.basename(img_path),
                "gt_embs": [emb.cpu().numpy().tolist() for emb in gt_embs],
                "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
                "gt_shapes": gt_shapes,
                "sam_shapes": sam_shapes,
            }

    def create_gt_masks(self, img_path: str) -> list[np.ndarray]:
        """Loads the JSON annotation next to the image and creates target masks."""
        size = Image.open(img_path).size
        mask_file = img_path.replace(".jpg", ".json")
        if not os.path.exists(mask_file):
            return []  # or raise an exception if you want

        with open(mask_file, "r") as f:
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
        """Remove SAM masks that overlap with any GT mask by IOU > 0.95."""
        return [
            mask for mask in sam_masks
            if not any(iou(mask, gt_mask) > 0.95 for gt_mask in gt_masks)
        ]

    def create_sam_masks(self, img_path: str) -> list[np.ndarray]:
        """Create SAM masks (as binary arrays) for the image at `img_path`."""
        res = self.sme.segment_path(img_path)
        # Convert to [0..255], as the code later expects an 8-bit mask
        sam_masks = [(mask["segmentation"].astype(np.uint8) * 255) for mask in res]
        return sam_masks

    def reshape_image(self, image: np.ndarray, size: int = 1024) -> np.ndarray:
        """Resize the image or mask to `size x size`."""
        return cv2.resize(image, (size, size))

    def get_shapes_from_masks(self, masks: list[np.ndarray]):
        """
        Extract contours from each mask and return them in a
        list-of-list-of-tuples format (similar to ReasonSeg).
        """
        shapes = [
            cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] 
            for mask in masks
        ]
        # Convert each contour into a list of (x, y) tuples
        return [[[tuple(p[0]) for p in contour] for contour in shape] for shape in shapes]

    def inference_preprocess(self, img_path: str, mask_only: bool) -> dict:
        """
        Preprocess an image for inference (no GT, only SAM).
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.reshape_image(img)
        img_pil = Image.fromarray(img)

        sam_masks = self.create_sam_masks(img_path)
        sam_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in sam_masks]
        sam_shapes = self.get_shapes_from_masks(sam_masks)

        return {
            "img": os.path.basename(img_path),
            "sam_shapes": sam_shapes,
            "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
        }


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError("Unknown type:", type(obj))


if __name__ == "__main__":
    # Load your pipeline configuration
    config = load_yaml_config("config.yaml")

    # ------------------------------------------------------------------
    # 1. Process ReasonSeg dataset
    # ------------------------------------------------------------------
    # base_image_dir = config.dataset.ReasonSeg.image_dir
    # dirs = ["train", "val", "test"]

    # # create the data directory if it doesn't exist
    # os.makedirs("data", exist_ok=True)

    # for d in dirs:
    #     print(f"Processing ReasonSeg {d}...")
    #     config.dataset.ReasonSeg.image_dir = os.path.join(base_image_dir, d)
    #     pipeline = PreprocessPipeline(config)

    #     res = pipeline.run_all(mask_only=False, dataset_name="ReasonSeg")
    #     out_file = f"data/{os.path.basename(config.dataset.ReasonSeg.image_dir)}_ReasonSeg.jsonl"

    #     with open(out_file, "w") as f:
    #         for r in res:
    #             f.write(json.dumps(r, default=default) + "\n")

    # ------------------------------------------------------------------
    # 2. Process COCO dataset
    # ------------------------------------------------------------------
    print("Processing COCO dataset...")
    fo.config.dataset_zoo_dir = config.dataset.COCO.dataset_zoo_dir

    pipeline = PreprocessPipeline(config)
    coco_res = pipeline.run_all(mask_only=False, dataset_name="COCO")

    # Save to JSON lines
    with open("data/COCO_train.jsonl", "w") as f:
        for r in coco_res:
            f.write(json.dumps(r, default=default) + "\n")

    print("All done!")
