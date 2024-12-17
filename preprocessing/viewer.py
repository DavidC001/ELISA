import json
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def load_train_data(train_json_path):
    """Load the train.json file."""
    with open(train_json_path, "r") as f:
        return json.load(f)


def load_annotation(file_path):
    """Load the JSON annotation file for an image."""
    with open(file_path, "r") as f:
        return json.load(f)


def draw_shapes(image, shapes):
    """Draw shapes on the image using Pillow."""
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for shape in shapes:
        # Ensure points are tuples of (x, y)
        points = [tuple(point) for point in shape["points"]]

        if shape["label"] == "target":
            color = (0, 255, 0, 100)  # Green with transparency
        elif shape["label"] == "ignore":
            color = (255, 0, 0, 0)  # Red with transparency
        elif shape["label"] == "flag":
            color = (0, 0, 255, 100)  # Blue with transparency
        else:
            color = (255, 255, 0, 50)  # Yellow with transparency

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


def display_image_with_annotations(image_path, annotation_path, query, outputs):
    """Display the image with annotations."""
    # Open the image
    image = Image.open(image_path).convert("RGBA")

    # Load annotation
    annotation = load_annotation(annotation_path)

    # Draw the shapes on the image
    annotated_image = draw_shapes(image.copy(), annotation["shapes"])

    if len(annotation["shapes"]) == 1:
        return
    # Display the image with annotations
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Query: {query}\nOutput: {outputs}", wrap=True, fontsize=10)
    plt.show()


def main():
    # Paths to the train.json file and the dataset directory
    train_json_path = "C:/Users/david/Documents/progetto/LISA/ReasonSeg/explanatory/train.json"
    train_dir = "C:/Users/david/Documents/progetto/LISA/ReasonSeg/train/"
    # Load train.json
    train_data = load_train_data(train_json_path)

    for entry in train_data:
        image_file = os.path.join(train_dir, entry["image"])
        json_file = os.path.join(train_dir, entry["json"])

        print(f"Displaying: {entry['image']} with annotations...")

        display_image_with_annotations(
            image_path=image_file,
            annotation_path=json_file,
            query=entry["query"],
            outputs=entry["outputs"],
        )
        # input("Press Enter to display the next image...")  # Wait for user input


if __name__ == "__main__":
    main()
