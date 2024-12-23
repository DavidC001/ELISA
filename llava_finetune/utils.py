import torch
import json
import os
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import wandb

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================
# 1. Dataset Definition
# ==========================
class CustomDataset(Dataset):
    def __init__(self, json_path, image_dir, exp_json_path=None, load_images=False, top_samples=30):
        """Initializes the CustomDataset class.

        Args:
            json_path (str): Path to the jsonl file containing the data.
            image_dir (str): Path to the directory containing the images.
            exp_json_path (str, optional): Path to the json file containing the explanatory data. Defaults to None.
            load_images (bool, optional): Whether to pre-load the images. Defaults to False.
        """
        self.image_dir = image_dir
        self.load_images = load_images
        self.top_samples = top_samples
        self.data = self.load_data(json_path, image_dir, exp_json_path)

    def load_data(self, json_path, image_dir, exp_json_path):
        data = []
        answers = {}
        if exp_json_path:
            with open(exp_json_path, "r") as f:
                exp_data = json.load(f)
            for sample in exp_data:
                image = sample["image"]
                answer = sample["outputs"]
                answers[image] = answer

        # compute the number of positive classes and negative classes
        gt_classes = 0
        sam_classes = 0
        with open(json_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                image = sample["img"]
                image_json_path = os.path.join(image_dir, image.split(".")[0] + ".json")
                with open(image_json_path, "r") as f2:
                    image_queries = json.load(f2)["text"]

                if (len(sample["gt_embs"]) == 0) or (len(sample["sam_embs"]) == 0):
                    continue
                
                gt_classes += len(sample["gt_embs"])
                sam_classes += len(sample["sam_embs"])
                
                data.append(
                    {
                        "image": (
                            Image.open(os.path.join(image_dir, image))
                            if self.load_images
                            else image
                        ),
                        "queries": image_queries,
                        "answer": answers.get(image, None),
                        "gt_embs": torch.tensor(sample["gt_embs"]).view(
                            len(sample["gt_embs"]), -1
                        ),
                        "gt_shapes": sample["gt_shapes"],
                        "sam_embs": torch.tensor(sample["sam_embs"][:self.top_samples]).view(
                            min(len(sample["sam_embs"]),self.top_samples), -1
                        ),
                        "sam_shapes": sample["sam_shapes"][:self.top_samples],
                    }
                )
        
        print(f"Number of positive classes: {gt_classes}")
        print(f"Number of negative classes: {sam_classes}")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = (
            sample["image"]
            if self.load_images
            else Image.open(os.path.join(self.image_dir, sample["image"]))
        )
        query = sample["queries"][torch.randint(0, len(sample["queries"]), (1,)).item()]
        return {
            "image": image,
            "image_path": os.path.join(self.image_dir, sample["image"]) if not self.load_images else None,
            "queries": query,
            "answer": sample["answer"],
            "gt_embs": sample["gt_embs"],
            "gt_shapes": sample["gt_shapes"],
            "sam_embs": sample["sam_embs"],
            "sam_shapes": sample["sam_shapes"],
        }

def collate_fn(batch):
    new_batch = {}
    for key in batch[0]:
        new_batch[key] = [sample[key] for sample in batch]
    return new_batch


def get_dataloaders(
    explanatory_train, image_dir, batch_size, train_jsonl, val_jsonl, test_jsonl
):
    """Get the training, validation, and test data loaders.

    Args:
        explanatory_train (_type_): file path to the explanatory data for training
        image_dir (_type_): path to the directory containing the images
        batch_size (_type_): batch size for the data loaders
        train_jsonl (_type_): jsonl file containing the training data masks
        val_jsonl (_type_): jsonl file containing the validation data masks
        test_jsonl (_type_): jsonl file containing the test data masks

    Returns:
        DataLoader: training data loader
        DataLoader: validation data loader
        DataLoader: test data loader
    """
    print("Loading Training Data")
    data_train = CustomDataset(
        json_path=train_jsonl,
        image_dir=os.path.join(image_dir, "train"),
        exp_json_path=explanatory_train,
    )

    print("Loading Validation Data")
    data_val = CustomDataset(
        json_path=val_jsonl, image_dir=os.path.join(image_dir, "val")
    )

    print("Loading Test Data")
    data_test = CustomDataset(
        json_path=test_jsonl, image_dir=os.path.join(image_dir, "test")
    )

    # Create DataLoaders
    data_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    data_val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    data_test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return data_loader, data_val_loader, data_test_loader


# ==========================
# 2. WandB Initialization
# ==========================
def initialize_wandb(exp_name, exp_config):
    """
    Initialize a wandb run for the experiment.

    Args:
        exp_name (str): Name of the experiment.
        exp_config (dict): Experiment-specific configuration.
    """
    wandb.init(
        project="LISA_ACV",
        name=exp_name + "_" + wandb.util.generate_id(),
        config=exp_config,
    )


# ==========================
# 3. Visualization Functions
# ==========================
def draw_shapes(
    image: Image.Image, 
    shapes: list[list[list[tuple]]], 
    resize: tuple = (1024, 1024), 
    enumerate_masks: bool = True,
    mask_names: str = None
) -> Image.Image:
    """
    Draw segmentation masks on an image with distinct colors and optional enumeration.

    Args:
        image (PIL.Image.Image): The original image.
        shapes (List[List[List[Tuple[int, int]]]]): 
            A list where each element represents a mask, which is a list of polygons, 
            and each polygon is a list of (x, y) tuples.
        resize (Tuple[int, int], optional): Desired image size. Defaults to (1024, 1024).
        enumerate_masks (bool, optional): If True, numbers each mask on the image. Defaults to True.
        mask_names (str, optional): List of mask names. Defaults to None.

    Returns:
        PIL.Image.Image: The image with drawn masks.
    """
    # Convert and resize the original image
    image = image.convert("RGBA").resize(resize)
    
    # Initialize Matplotlib's tab20 colormap for distinct colors
    cmap = plt.get_cmap('tab20')
    num_masks = len(shapes)
    colors = [mcolors.to_rgba(cmap(i % 20), alpha=0.4) for i in range(num_masks)]
    
    # Convert RGBA floats to integers (0-255)
    colors = [tuple(int(c * 255) for c in color) for color in colors]
    
    # Create an overlay for drawing masks
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Attempt to load a TrueType font; fallback to default if unavailable
    font = ImageFont.load_default(20)

    for i, mask in enumerate(shapes):
        color = colors[i]
        all_points = []

        for polygon in mask:
            points = [tuple(point) for point in polygon]
            if len(points) < 2:
                print(f"Warning: Polygon {polygon} in mask {i} has less than 2 points and will be skipped.")
                continue

            # Draw the filled polygon with transparency
            draw.polygon(points, fill=color)
            # Draw the polygon outline
            draw.line(points + [points[0]], fill=(0, 0, 0, 255), width=2)
            # Collect points to compute centroid later
            all_points.extend(points)
        
        if enumerate_masks and all_points:
            # Calculate centroid of all points in the mask for label placement
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            text = str(i + 1)  if not mask_names else mask_names[i]

            # Draw white outline for better text visibility
            draw.text((centroid[0]+1, centroid[1]+1), text, font=font, fill="white")
            draw.text((centroid[0]-1, centroid[1]-1), text, font=font, fill="white")
            draw.text((centroid[0]+1, centroid[1]-1), text, font=font, fill="white")
            draw.text((centroid[0]-1, centroid[1]+1), text, font=font, fill="white")
            # Draw the black enumeration number
            draw.text(centroid, text, font=font, fill="black")

    # Combine the overlay with the original image
    combined = Image.alpha_composite(image, overlay)
    return combined