
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "classification"
OUTPUT_DIR = ROOT / "outputs" / "resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "best_resnet18.pth"
LAST_CHECKPOINT = OUTPUT_DIR / "last_resnet18.pth"
HISTORY_FILE = OUTPUT_DIR / "training_history.csv"

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Resume from last checkpoint if one exists
RESUME = True


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


# ============================================================
# 4. TRAINING AND VALIDATION
# ============================================================

def run_epoch(model, loader, criterion, device, optimizer=None):

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# ============================================================
# 5. MAIN TRAINING PIPELINE
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------------
    # Load classification datasets
    # --------------------------------------------------------

    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=eval_transform,
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation classes differ.")

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Load pretrained ResNet-18
    # --------------------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace original 1000-class classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    labels = torch.tensor(train_dataset.targets)

    counts = torch.bincount(
        labels,
        minlength=num_classes,
    ).float()

    class_weights = counts.sum() / (
        num_classes * counts.clamp(min=1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Checkpoint initialization
    # --------------------------------------------------------

    history = []
    best_val_accuracy = -1.0
    start_epoch = 0

    if RESUME and LAST_CHECKPOINT.exists():

        checkpoint = torch.load(
            LAST_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )

        if checkpoint["class_names"] != class_names:
            raise ValueError(
                "Checkpoint classes do not match dataset."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        history = checkpoint["history"]

        # Restore RNG state for more consistent continuation
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

        if (
            device.type == "cuda"
            and "cuda_rng_state" in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )

        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])

        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        print(
            f"\nResuming training from epoch {start_epoch + 1}"
        )

    else:
        print("\nStarting fresh training")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, EPOCHS):

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}",
            flush=True,
        )

        # ----------------------------------------------------
        # Record training history
        # ----------------------------------------------------

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
            }, BEST_CHECKPOINT)

            print(
                f"  New best model saved! "
                f"Validation accuracy: {val_accuracy:.4f}"
            )

        # ----------------------------------------------------
        # Save resumable checkpoint
        # ----------------------------------------------------

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "epoch": epoch + 1,
            "best_val_accuracy": best_val_accuracy,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else []
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }, LAST_CHECKPOINT)

        # Update history CSV after each epoch
        pd.DataFrame(history).to_csv(
            HISTORY_FILE,
            index=False,
        )

        print("  Checkpoint saved.", flush=True)

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model: {BEST_CHECKPOINT}")
    print(f"Last checkpoint: {LAST_CHECKPOINT}")
    print(f"Training history: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
