
from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[2]

    model_path = root / "outputs/yolo/ppe_training/weights/best.pt"
    data_path = root / "data/dataset.yaml"
    output_dir = root / "outputs/yolo"

    # Load our trained model
    model = YOLO(str(model_path))

    # Evaluate on the test dataset
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(output_dir),
        name="test_evaluation",
        plots=True,
    )

    # Print overall results
    print("\n--- YOLO TEST RESULTS ---")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    # Save per-class metrics
    rows = []

    for class_id, class_name in model.names.items():
        class_results = metrics.class_result(class_id)

        rows.append({
            "class": class_name,
            "precision": class_results[0],
            "recall": class_results[1],
            "map50": class_results[2],
            "map50_95": class_results[3],
        })

    df = pd.DataFrame(rows)

    save_path = output_dir / "test_metrics.csv"
    df.to_csv(save_path, index=False)

    print(f"\nMetrics saved to: {save_path}")

    # Generate predictions on held-out test images
    model.predict(
        source=str(root / "data/images/test"),
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        project=str(output_dir),
        name="test_predictions",
    )

    print("\nPrediction images saved successfully.")


if __name__ == "__main__":
    main()
