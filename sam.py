import glob
import os

import cv2
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

config = dotenv_values(".env")

models = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

# Download the model if it doesn't exist
chosen_model = config.get("SAM_MODEL")
checkpoint_name = models[chosen_model].split("/")[-1]

print(f"Using model: {chosen_model}, with checkpoint: {checkpoint_name}")

checkpoint_dir = os.path.expanduser(config.get("SAM_CHECKPOINTS_DIR"))
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

if not os.path.exists(checkpoint_path):
    print("Downloading model...")
    import urllib.request

    urllib.request.urlretrieve(models[chosen_model], checkpoint_path)

else:
    print("Model has already been downloaded.")

sam = sam_model_registry[chosen_model](checkpoint=checkpoint_path)

image_dir = config.get("IMAGE_DIR")

mask_generator = SamAutomaticMaskGenerator(sam)

for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    masks = mask_generator.generate(image_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Found {len(masks)} masks for {image_path}")

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")

    for mask in masks:
        plt.imshow(mask, alpha=0.5)

    plt.show()
