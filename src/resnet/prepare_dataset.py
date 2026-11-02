
from pathlib import Path
from collections import Counter

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "classification"

# PPE-related classes used for classification
SELECTED_CLASSES = {
    0, 1, 2, 3, 4, 7, 8, 9, 10
}

with open(DATA_DIR / "dataset.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CLASS_NAMES = config["names"]


def create_crops(split):
    image_dir = DATA_DIR / "images" / split
    label_dir = DATA_DIR / "labels" / split

    counts = Counter()

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        with Image.open(image_path) as original:
            image = original.convert("RGB")

        width, height = image.size

        for index, line in enumerate(label_path.read_text().splitlines()):
            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            if class_id not in SELECTED_CLASSES:
                continue

            xc, yc, bw, bh = map(float, values[1:])

            # Convert normalized bounding box to pixels
            x1 = max(0, int((xc - bw / 2) * width))
            y1 = max(0, int((yc - bh / 2) * height))
            x2 = min(width, int((xc + bw / 2) * width))
            y2 = min(height, int((yc + bh / 2) * height))

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract the object
            crop = image.crop((x1, y1, x2, y2))

            class_name = CLASS_NAMES[class_id]
            save_dir = OUTPUT_DIR / split / class_name
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{image_path.stem}_obj{index}.jpg"
            crop.save(save_dir / filename, quality=95)

            counts[class_name] += 1

    print(f"\n{split.upper()} CROPS")
    for name in sorted(counts):
        print(f"{name}: {counts[name]}")

    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        create_crops(split)

    print("\nClassification dataset created successfully!")
