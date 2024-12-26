import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from configuration import ProjectConfig, dataclass, load_yaml_config
from preprocessing.alphaclip import AlphaCLIPEncoder
from preprocessing.sam import SegmentationMaskExtractor


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
        
        # We'll store the COCO categories here once we load them.
        self.coco_cat_map = {}

    def run_all(self, mask_only: bool = False, dataset_name: str = "ReasonSeg") -> list[dict]:
        """
        Runs the pipeline on either the ReasonSeg dataset or the COCO dataset.
        """
        res = []

        if dataset_name == "ReasonSeg":
            for image in tqdm(glob.glob(self.ReasonSeg_dataset.image_dir + "/*.jpg")):
                try:
                    res.append(self.run_ReasonSeg(image, mask_only))
                except Exception as e:
                    print(f"Error in {image}: {e}")

        else:
            # COCO dataset
            json_labels_path = os.path.join(
                self.COCO_dataset.dataset_zoo_dir,
                self.COCO_dataset.name,
                "train",
                "labels.json"
            )
            with open(json_labels_path, "r") as f:
                labels = json.load(f)

            # Build a category-id -> category-name map
            self.coco_cat_map = {
                cat["id"]: cat["name"]
                for cat in labels["categories"]
            }

            # Make a dict to quickly look up image info by ID
            images_dict = {img["id"]: img for img in labels["images"]}

            # Group annotations by image_id
            ann_by_img_id = {}
            for ann in labels["annotations"]:
                img_id = ann["image_id"]
                ann_by_img_id.setdefault(img_id, []).append(ann)

            # Process each image
            for img_id, img_info in tqdm(images_dict.items()):
                try:
                    image_path = os.path.join(
                        self.COCO_dataset.dataset_zoo_dir,
                        self.COCO_dataset.name,
                        "train",
                        "data",
                        img_info["file_name"]
                    )

                    # The annotations for this image
                    annotations = ann_by_img_id.get(img_id, [])

                    out = self.run_COCO(image_path, img_info, annotations, mask_only)
                    
                    if out is not None:
                        for i in range(len(out["gt_embs"])):
                            res.append({
                                "img": out["img"],
                                "gt_embs": [out["gt_embs"][i]],
                                "sam_embs": out["sam_embs"],
                                "gt_shapes": [out["gt_shapes"][i]],
                                "sam_shapes": out["sam_shapes"],
                                "gt_labels": [out["gt_labels"][i]],
                            })
                except Exception as e:
                    print(f"Error processing {img_info['file_name']}: {e}")

        return res

    def run_COCO(self, image_path: str, img_info: dict, annotations: list[dict], mask_only: bool) -> dict:
        """
        Run the preprocessing pipeline on a COCO sample, using polygon-based segmentations.
        
        We'll create GT masks from the 'segmentation' polygons.
        We'll also store the category name for each mask, so we can build text prompts later.
        """
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None

        # Attempt to load the image size from 'img_info'
        w = img_info.get("width")
        h = img_info.get("height")

        # If not present, fallback to reading the image
        if w is None or h is None:
            try:
                pil_img_for_size = Image.open(image_path)
                w, h = pil_img_for_size.size
            except:
                print(f"Could not open image: {image_path}")
                return None

        # ------------------------------------------------------------------
        # 1) Create GT masks from polygon segmentations
        # ------------------------------------------------------------------
        gt_masks = []
        gt_labels = []  # keep track of category names
        for ann in annotations:
            if ann.get("iscrowd", 0) == 1:
                # skip crowd annotations
                continue

            segm = ann.get("segmentation", None)
            if not segm:
                # skip empty segmentation
                continue

            # Create a new empty PIL mask
            mask_pil = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_pil)

            # In COCO, 'segmentation' can be multiple polygons, each a list [x1,y1, x2,y2, ...]
            if isinstance(segm, list):
                for polygon in segm:
                    if len(polygon) < 6:  # need at least 3 points
                        continue
                    xy_points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(xy_points, fill=255)
            
            # Convert that mask to NumPy
            gt_masks.append(np.array(mask_pil))

            # Also store the category name
            cat_id = ann["category_id"]
            cat_name = self.coco_cat_map.get(cat_id, "unknown")
            gt_labels.append(cat_name)

        # ------------------------------------------------------------------
        # 2) Create SAM masks
        # ------------------------------------------------------------------
        sam_masks = self.create_sam_masks(image_path)

        # ------------------------------------------------------------------
        # 3) Read image, reshape to 1024 x 1024
        # ------------------------------------------------------------------
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image with cv2: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.reshape_image(img)  # -> 1024 x 1024
        img_pil = Image.fromarray(img)

        # ------------------------------------------------------------------
        # 4) Reshape all masks to 1024 x 1024
        # ------------------------------------------------------------------
        gt_masks = [self.reshape_image(m) for m in gt_masks]
        sam_masks = [self.reshape_image(m) for m in sam_masks]

        # ------------------------------------------------------------------
        # 5) Remove SAM masks that significantly overlap with GT
        # ------------------------------------------------------------------
        sam_masks = self.remove_gt_masks(sam_masks, gt_masks)

        # ------------------------------------------------------------------
        # 6) Get shapes from the final masks
        # ------------------------------------------------------------------
        gt_shapes = self.get_shapes_from_masks(gt_masks)
        sam_shapes = self.get_shapes_from_masks(sam_masks)

        # ------------------------------------------------------------------
        # 7) Compute embeddings
        # ------------------------------------------------------------------
        with torch.no_grad():
            gt_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in gt_masks]
            sam_embs = [self.ace.get_visual_embedding(img_pil, mask, mask_only) for mask in sam_masks]

        return {
            "img": os.path.basename(image_path),
            "gt_embs": [emb.cpu().numpy().tolist() for emb in gt_embs],
            "sam_embs": [emb.cpu().numpy().tolist() for emb in sam_embs],
            "gt_shapes": gt_shapes,
            "sam_shapes": sam_shapes,
            "gt_labels": gt_labels,  # store array of category names
        }

    def run_ReasonSeg(self, img_path: str, mask_only: bool) -> dict:
        """
        Run the preprocessing pipeline on a ReasonSeg image.
        (No changes from your original approach.)
        """
        with torch.no_grad():
            # Create GT masks from the JSON annotation next to the image
            gt_masks = self.create_gt_masks(img_path)
            # Create SAM masks
            sam_masks = self.create_sam_masks(img_path)

            # Read and reshape the image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.reshape_image(img)
            img_pil = Image.fromarray(img)

            # Resize masks
            gt_masks = [self.reshape_image(mask) for mask in gt_masks]
            sam_masks = [self.reshape_image(mask) for mask in sam_masks]

            # Remove overlapping SAM masks
            sam_masks = self.remove_gt_masks(sam_masks, gt_masks)

            # Shapes
            gt_shapes = self.get_shapes_from_masks(gt_masks)
            sam_shapes = self.get_shapes_from_masks(sam_masks)

            # Embeddings
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
        """
        Loads the JSON annotation next to the image and creates target masks.
        (Used by ReasonSeg only.)
        """
        size = Image.open(img_path).size
        mask_file = img_path.replace(".jpg", ".json")
        if not os.path.exists(mask_file):
            return []

        with open(mask_file, "r") as f:
            data = json.load(f)

        # ReasonSeg label "target" -> polygon points
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
        """
        Remove any SAM masks that overlap with any GT mask by IOU > 0.95.
        """
        return [
            mask for mask in sam_masks
            if not any(iou(mask, gt_mask) > 0.95 for gt_mask in gt_masks)
        ]

    def create_sam_masks(self, img_path: str) -> list[np.ndarray]:
        """
        Create SAM masks (binary arrays) for the image at `img_path`.
        """
        res = self.sme.segment_path(img_path)
        # Convert to [0..255], as the code later expects an 8-bit mask
        sam_masks = [(mask["segmentation"].astype(np.uint8) * 255) for mask in res]
        return sam_masks

    def reshape_image(self, image: np.ndarray, size: int = 1024) -> np.ndarray:
        """
        Resize the image or mask to `size x size`.
        """
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
        # Convert each contour to a list of (x, y) tuples
        return [
            [[tuple(p[0]) for p in contour] for contour in shape]
            for shape in shapes
        ]

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
    """
    Helper for JSON serialization of NumPy arrays.
    """
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
    # Example: Process ReasonSeg dataset
    # ------------------------------------------------------------------
    print("Processing ReasonSeg dataset...")
    base_image_dir = config.dataset.ReasonSeg.image_dir
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Split: {split}")
        config.dataset.ReasonSeg.image_dir = os.path.join(base_image_dir, split)
        pipeline = PreprocessPipeline(config)

        reasonseg_res = pipeline.run_all(mask_only=False, dataset_name="ReasonSeg")

        # Save results (e.g., JSONL)
        out_dir = "data"
        os.makedirs(out_dir, exist_ok=True)
        out_file = f"{out_dir}/ReasonSeg_{split}.jsonl"
        with open(out_file, "w") as f:
            for r in reasonseg_res:
                f.write(json.dumps(r, default=default) + "\n")

    # ------------------------------------------------------------------
    # Example: Process COCO dataset
    # ------------------------------------------------------------------
    print("\nProcessing COCO dataset...")
    pipeline = PreprocessPipeline(config)
    coco_res = pipeline.run_all(mask_only=False, dataset_name="COCO")

    # Save results
    with open("data/COCO_train.jsonl", "w") as f:
        for r in coco_res:
            f.write(json.dumps(r, default=default) + "\n")

    print("All done!")
