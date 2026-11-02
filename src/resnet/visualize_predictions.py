
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/resnet"

# Load saved predictions and confidence scores
df = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")

correct = df[df["correct"] == True]
incorrect = df[df["correct"] == False]

# Select reproducible examples
correct_samples = correct.sample(
    n=min(6, len(correct)),
    random_state=42,
)

incorrect_samples = incorrect.sample(
    n=min(6, len(incorrect)),
    random_state=42,
)

samples = pd.concat(
    [correct_samples, incorrect_samples],
    ignore_index=True,
)

# Create visualization
fig, axes = plt.subplots(3, 4, figsize=(15, 11))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):

    image_path = ROOT / row["image"]

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    is_correct = bool(row["correct"])

    color = "green" if is_correct else "red"

    ax.set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}\n"
        f"Confidence: {row['confidence']:.2%}",
        color=color,
        fontsize=10,
    )

    ax.axis("off")

plt.suptitle(
    "ResNet-18 PPE Classification Predictions",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()

save_path = OUTPUT_DIR / "prediction_confidence_samples.png"

plt.savefig(save_path, dpi=200)
plt.show()

print(f"Visualization saved to: {save_path}")
