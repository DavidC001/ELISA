import glob
import random

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from configuration import load_yaml_config
from preprocess import PreprocessPipeline


def draw_shapes(image: Image, shapes: list[list]) -> Image:
    """Draw shapes on the image using Pillow."""
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for shape in shapes:
        for s in shape:
            # Ensure points are tuples of (x, y)
            points = [tuple(point) for point in s]
            if len(points) < 2:
                print(f"Shape {s} has less than 2 points")
                continue

            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                100,
            )  # Random color with transparency

            # Draw the polygon on the overlay
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill=(0, 0, 0, 255), width=2)  # Black outline

            # check if points are outside the image size
            for point in points:
                if point[0] > image.size[0] or point[1] > image.size[1]:
                    print(f"Point {point} is outside the image size {image.size}")

    # Combine the overlay with the original image
    combined = Image.alpha_composite(image, overlay)
    return combined


config = load_yaml_config("config.yaml")

pp = PreprocessPipeline(config)

img_path = glob.glob(pp.dataset.image_dir + "/*.jpg")[0]

res = pp.inference_preprocess(img_path)

image = Image.open(img_path).convert("RGBA")
image = image.resize((config.sam.resize, config.sam.resize))

comb = draw_shapes(image, res["sam_shapes"])


plt.figure(figsize=(12, 8))
plt.imshow(comb)
plt.axis("off")
plt.savefig("output/o.png")
