
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL = ROOT / "outputs/yolo/ppe_training/weights/best.pt"
RESNET_PREDICTIONS = ROOT / "outputs/resnet/test_predictions.csv"
OUTPUT_DIR = ROOT / "outputs/comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PPE_CLASSES = [
    "boots", "gloves", "goggles", "helmet",
    "no_boots", "no_gloves", "no_goggle",
    "no_helmet", "vest",
]

BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.01


def main():
    # 1. Load YOLO and ResNet's existing test records
    model = YOLO(str(YOLO_MODEL))
    df = pd.read_csv(RESNET_PREDICTIONS)

    if not df["actual_class"].isin(PPE_CLASSES).all():
        raise ValueError("Unexpected class in test data.")

    # Map PPE class names to YOLO's numeric IDs
    ppe_ids = [
        class_id
        for class_id, name in model.names.items()
        if name in PPE_CLASSES
    ]

    paths = [str(ROOT / p) for p in df["image"]]

    predictions = []
    confidences = []

    # 2. Predict on exactly the same test crops
    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]

        results = model.predict(
            source=batch_paths,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            classes=ppe_ids,
            device=0,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                predictions.append("no_detection")
                confidences.append(0.0)
                continue

            # Select highest-confidence PPE detection
            best_index = int(boxes.conf.argmax().item())

            class_id = int(
                boxes.cls[best_index].item()
            )

            confidence = float(
                boxes.conf[best_index].item()
            )

            predictions.append(model.names[class_id])
            confidences.append(confidence)

    # 3. Store predictions
    df["yolo_predicted_class"] = predictions
    df["yolo_confidence"] = confidences
    df["yolo_correct"] = (
        df["actual_class"] == df["yolo_predicted_class"]
    )

    df.to_csv(
        OUTPUT_DIR / "yolo_crop_predictions.csv",
        index=False,
    )

    # 4. Calculate classification metrics
    y_true = df["actual_class"]
    y_pred = df["yolo_predicted_class"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=PPE_CLASSES,
            average="macro",
            zero_division=0,
        )
    )

    print("\n--- YOLO CROP CLASSIFICATION RESULTS ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")

    print(classification_report(
        y_true,
        y_pred,
        labels=PPE_CLASSES,
        zero_division=0,
    ))

    print(
        "Crops without detection:",
        (df["yolo_predicted_class"] == "no_detection").sum()
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "yolo_crop_predictions.csv"
    )


if __name__ == "__main__":
    main()
