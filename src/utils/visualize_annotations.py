
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import yaml

# 1. Define project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

IMAGES_DIR = DATA_DIR / "images" / "train"
LABELS_DIR = DATA_DIR / "labels" / "train"

# 2. Load class names from the dataset configuration
with open(DATA_DIR / "dataset.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CLASS_NAMES = config["names"]

# 3. Collect training images
images = sorted(
    image
    for image in IMAGES_DIR.iterdir()
    if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

random.seed(42)
selected_images = random.sample(images, min(4, len(images)))

# 4. Create a grid to display sample images
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, image_path in zip(axes.flat, selected_images):

    # Load the image and convert BGR to RGB
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    # Find the corresponding annotation file
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        with open(label_path, encoding="utf-8") as file:
            annotations = file.readlines()

        for annotation in annotations:
            values = annotation.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(
                float, values[1:]
            )

            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            class_name = CLASS_NAMES[class_id]

            # Draw bounding box and class name
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                image,
                class_name,
                (max(0, x1), max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    ax.imshow(image)
    ax.set_title(image_path.name)
    ax.axis("off")

plt.tight_layout()

# 5. Save the visualization
output_path = ROOT / "outputs" / "comparison" / "dataset_samples.png"
plt.savefig(output_path, dpi=150)
plt.show()

print(f"Visualization saved to: {output_path}")
