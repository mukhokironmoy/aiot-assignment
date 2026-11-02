
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/comparison"

YOLO_CSV = OUTPUT_DIR / "yolo_crop_predictions.csv"
RESNET_CSV = ROOT / "outputs/resnet/test_predictions.csv"

CLASSES = [
    "boots", "gloves", "goggles", "helmet",
    "no_boots", "no_gloves", "no_goggle",
    "no_helmet", "vest",
]


def calculate_metrics(actual, predicted):

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():

    yolo = pd.read_csv(YOLO_CSV)
    resnet = pd.read_csv(RESNET_CSV)

    # Verify that both predictions refer to identical crops
    if not yolo["image"].equals(resnet["image"]):
        raise ValueError("Test images are not aligned.")

    if not yolo["actual_class"].equals(resnet["actual_class"]):
        raise ValueError("Ground-truth labels do not match.")

    yolo_metrics = calculate_metrics(
        yolo["actual_class"],
        yolo["yolo_predicted_class"],
    )

    resnet_metrics = calculate_metrics(
        resnet["actual_class"],
        resnet["predicted_class"],
    )

    comparison = pd.DataFrame({
        "YOLO11n": yolo_metrics,
        "ResNet18": resnet_metrics,
    })

    comparison.to_csv(
        OUTPUT_DIR / "classification_comparison.csv"
    )

    print("\n--- CLASSIFICATION COMPARISON ---")
    print(comparison.round(4))

    print("\n--- MODEL CHARACTERISTICS ---")

    details = pd.DataFrame([
        {
            "Model": "YOLO11n",
            "Primary Objective": "Object Detection",
            "Input": "Full images",
            "Output": "Bounding boxes + classes + confidence",
            "Multiple Objects": "Yes",
            "Localization": "Yes",
            "Epochs": 20,
            "Training Image Size": 640,
            "Training Batch Size": 8,
        },
        {
            "Model": "ResNet18",
            "Primary Objective": "Image Classification",
            "Input": "Object crops",
            "Output": "Class + confidence",
            "Multiple Objects": "No (one crop per prediction)",
            "Localization": "No",
            "Epochs": 20,
            "Training Image Size": 224,
            "Training Batch Size": 32,
        },
    ])

    details.to_csv(
        OUTPUT_DIR / "model_characteristics.csv",
        index=False,
    )

    print(details.to_string(index=False))

    print("\nComparison files saved successfully.")


if __name__ == "__main__":
    main()
