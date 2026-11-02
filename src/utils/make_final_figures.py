"""Generate report-ready figures from local project metrics (no retraining)."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# The comparison CSV is produced by src/utils/compare_models.py.
metrics = pd.read_csv(OUT / "classification_comparison.csv", index_col=0)
labels = ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]
keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
x = np.arange(len(keys))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width / 2, metrics.loc[keys, "YOLO11n"] * 100, width, label="YOLO11n (detection adapted to crop labels)")
b2 = ax.bar(x + width / 2, metrics.loc[keys, "ResNet18"] * 100, width, label="ResNet-18")
ax.set_xticks(x, labels)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)
ax.set_title("Classification on the same 950 labelled PPE crops")
ax.legend(loc="upper left", fontsize=8)
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)
ax.grid(axis="y", alpha=0.2)
fig.text(0.02, 0.01, "YOLO uses highest-confidence PPE detection; no detections count as incorrect. Not a native-task comparison.", fontsize=8)
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig(OUT / "final_classification_comparison.png", dpi=200)
plt.close(fig)

bench = pd.read_csv(OUT / "inference_benchmark.csv").set_index("model")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
names = ["YOLO11n", "ResNet18"]
values = [float(bench.loc[n, "avg_inference_ms"]) for n in names]
bars = ax.bar(names, values)
ax.bar_label(bars, fmt="%.2f ms", padding=3)
ax.set_ylabel("Mean time per crop (ms) — lower is better")
ax.set_title("RTX 4050 single-crop inference (100 crops; batch size 1)")
ax.set_ylim(0, max(values) * 1.22)
ax.grid(axis="y", alpha=0.2)
fig.text(0.02, 0.01, "Excludes model load and disk I/O; ResNet requires external crop localization in real deployments.", fontsize=8)
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.savefig(OUT / "final_inference_comparison.png", dpi=200)
plt.close(fig)

# Generate a side-by-side error inspection image from aligned prediction rows.
yolo = pd.read_csv(OUT / "yolo_crop_predictions.csv")
resnet = pd.read_csv(ROOT / "outputs" / "resnet" / "test_predictions.csv")
if not yolo["image"].equals(resnet["image"]) or not yolo["actual_class"].equals(resnet["actual_class"]):
    raise ValueError("YOLO and ResNet rows are not aligned")
wrong = yolo[(yolo["yolo_predicted_class"] != yolo["actual_class"]) | (resnet["predicted_class"] != resnet["actual_class"])].copy()
wrong["resnet_predicted_class"] = resnet.loc[wrong.index, "predicted_class"]
wrong["resnet_confidence"] = resnet.loc[wrong.index, "confidence"]
# Prefer cases where ResNet is correct but adapted YOLO is wrong, sampled reproducibly.
subset = wrong[(wrong["resnet_predicted_class"] == wrong["actual_class"]) & (wrong["yolo_predicted_class"] != wrong["actual_class"])]
if len(subset) < 6:
    subset = wrong
subset = subset.sample(n=min(6, len(subset)), random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax in axes.flat:
    ax.axis("off")
for ax, (_, row) in zip(axes.flat, subset.iterrows()):
    img = plt.imread(ROOT / row["image"])
    ax.imshow(img)
    ax.set_title(f"True: {row['actual_class']}\nYOLO: {row['yolo_predicted_class']} ({row['yolo_confidence']:.0%})\nResNet: {row['resnet_predicted_class']} ({row['resnet_confidence']:.0%})", fontsize=9)
fig.suptitle("Same crops: detection-adapted YOLO vs ResNet classification", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "final_error_examples.png", dpi=200)
plt.close(fig)
print("Saved:")
for name in ["final_classification_comparison.png", "final_inference_comparison.png", "final_error_examples.png"]:
    print(OUT / name)
