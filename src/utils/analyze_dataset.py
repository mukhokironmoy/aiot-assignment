
from pathlib import Path
from collections import Counter

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

with open(DATA / "dataset.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

classes = config["names"]
results = []

for split in ["train", "val", "test"]:
    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split

    images = list(image_dir.glob("*.jpg"))
    images += list(image_dir.glob("*.png"))
    images += list(image_dir.glob("*.jpeg"))

    counts = Counter()
    invalid_lines = 0

    for image in images:
        label_file = label_dir / f"{image.stem}.txt"

        if not label_file.exists():
            continue

        for line in label_file.read_text().splitlines():
            parts = line.strip().split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                valid = (
                    class_id in classes
                    and all(0 <= v <= 1 for v in coords)
                    and coords[2] > 0
                    and coords[3] > 0
                )
            except ValueError:
                valid = False

            if not valid:
                invalid_lines += 1
                continue

            counts[class_id] += 1

    print(
        f"{split}: {len(images)} images, "
        f"{sum(counts.values())} objects, "
        f"{invalid_lines} invalid annotations"
    )

    for class_id, class_name in classes.items():
        results.append({
            "split": split,
            "class": class_name,
            "count": counts[class_id]
        })

df = pd.DataFrame(results)
table = df.pivot(index="class", columns="split", values="count")
table = table[["train", "val", "test"]]
table["total"] = table.sum(axis=1)

print("\nObject counts per class:")
print(table.to_string())

output = ROOT / "outputs" / "comparison" / "dataset_distribution.csv"
table.to_csv(output)

print(f"\nSaved to: {output}")
