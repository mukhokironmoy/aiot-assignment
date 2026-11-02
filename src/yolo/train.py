
from pathlib import Path
from ultralytics import YOLO


def main():
    # Locate the project root
    root = Path(__file__).resolve().parents[2]

    # Load a pretrained lightweight YOLO model
    model = YOLO("yolo11n.pt")

    # Fine-tune on our Construction-PPE dataset
    model.train(
        data=str(root / "data" / "dataset.yaml"),
        epochs=20,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=5,
        project=str(root / "outputs" / "yolo"),
        name="ppe_training",
        plots=True,
    )


if __name__ == "__main__":
    main()
