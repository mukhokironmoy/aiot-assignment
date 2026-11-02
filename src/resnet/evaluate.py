
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/classification/test"
OUTPUT_DIR = ROOT / "outputs/resnet"
CHECKPOINT = OUTPUT_DIR / "best_resnet18.pth"

BATCH_SIZE = 32


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # 1. Load the trained checkpoint
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # 2. Apply the same evaluation preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=transform,
    )

    if test_dataset.classes != class_names:
        raise ValueError("Test classes do not match checkpoint.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 3. Reconstruct ResNet-18
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 4. Generate predictions
    y_true = []
    y_pred = []
    confidences = []
    records = []

    print("\nRunning predictions...")

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs, dim=1
            )

            confidence, predictions = probabilities.max(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    # 5. Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    print("\n--- RESNET TEST RESULTS ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    # 6. Per-class classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
    )

    # 7. Save individual predictions
    for index, (image_path, _) in enumerate(test_dataset.samples):

        records.append({
            "image": str(Path(image_path).relative_to(ROOT)),
            "actual_class": class_names[y_true[index]],
            "predicted_class": class_names[y_pred[index]],
            "confidence": confidences[index],
            "correct": y_true[index] == y_pred[index],
        })

    predictions_df = pd.DataFrame(records)

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # 8. Generate confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("ResNet-18 Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # 9. Save classification errors
    errors = predictions_df[
        predictions_df["correct"] == False
    ].copy()

    errors = errors.sort_values(
        "confidence",
        ascending=False,
    )

    errors.to_csv(
        OUTPUT_DIR / "classification_errors.csv",
        index=False,
    )

    # 10. Display examples of incorrect predictions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax in axes.flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes.flat, errors.head(6).iterrows()):

        image = plt.imread(ROOT / row["image"])

        ax.imshow(image)
        ax.set_title(
            f"Actual: {row['actual_class']}\n"
            f"Predicted: {row['predicted_class']}\n"
            f"Confidence: {row['confidence']:.2%}",
            fontsize=10,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "classification_errors.png",
        dpi=200,
    )
    plt.close()

    print("\nEvaluation completed!")
    print(f"Total test crops: {len(y_true)}")
    print(f"Correct predictions: {sum(np.array(y_true) == np.array(y_pred))}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
