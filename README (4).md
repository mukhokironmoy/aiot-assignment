# Construction PPE Monitoring: YOLO11n vs ResNet-18

A reproducible comparison of **object detection** (YOLO11n) and **object-crop classification** (ResNet-18) on the [Ultralytics Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/).

> **Research finding:** ResNet-18 classifies *pre-extracted, ground-truth PPE crops* well; YOLO11n can locate multiple objects in full construction-site images, but this trained detector has low recall for missing-PPE classes. These tasks and their native metrics are **not interchangeable**. This prototype is intended to assist safety officers, not autonomously certify compliance.

## 1. Problem and objectives

The project investigates PPE compliance monitoring for construction-site images by:

1. Fine-tuning YOLO11n to detect and localize PPE and potential violations.
2. Fine-tuning ResNet-18 to classify individual PPE object crops.
3. Evaluating the detector using precision, recall, mAP@0.5, and mAP@0.5:0.95.
4. Evaluating the classifier using accuracy, precision, recall, F1, and a confusion matrix.
5. Comparing both models' classification behavior on the **same 950 labelled test crops** and benchmarking their single-crop inference on the same GPU.

## 2. Environment

- Windows 11, Python 3.12.10
- NVIDIA GeForce RTX 4050 Laptop GPU (approximately 6 GB VRAM)
- PyTorch 2.9.0+cu128, Torchvision 0.24.0+cu128, Ultralytics 8.4.184
- Other packages: Pillow, OpenCV, NumPy, pandas, scikit-learn, matplotlib, seaborn, PyYAML

Set up in PowerShell (with compatible GPU-enabled PyTorch already installed):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` is the dependency snapshot generated from the project's working environment. For a new GPU system, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to ensure a compatible CUDA build.

## 3. Data

Dataset: `https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip`

| Split | Images | All labelled boxes | Nine selected PPE boxes |
|---|---:|---:|---:|
| Train | 1,132 | 9,098 | 6,677 |
| Validation | 143 | 1,172 | 852 |
| Test | 141 | 1,251 | 950 |
| **Total** | **1,416** | **11,521** | **8,479** |

All 11 source classes: `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`. YOLO uses all 11. ResNet's crop experiment uses the nine PPE-specific categories, excluding `Person` and the ambiguous `none` label. The same original image splits are preserved before cropping, reducing image-level leakage.

Bounding boxes are represented in normalized YOLO format: `class_id x_center y_center width height`. A basic parser checked annotation field counts, class IDs, and coordinate ranges and reported zero invalid annotations; this does **not** guarantee semantic label quality.

Extract source images to `data/images/{train,val,test}` and annotations to `data/labels/{train,val,test}`. Ensure `data/dataset.yaml` points to the actual local `data` directory rather than the archive's default `construction-ppe` path.

## 4. Project layout

```text
ppe_yolo_vs_resnet/
├── data/
│   ├── dataset.yaml
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── classification/{train,val,test}/{class}/
├── src/
│   ├── yolo/{train,evaluate,evaluate_crops}.py
│   ├── resnet/{prepare_dataset,train,evaluate,visualize_predictions}.py
│   └── utils/{visualize_annotations,analyze_dataset,compare_models,benchmark_models,make_final_figures}.py
├── outputs/
│   ├── yolo/
│   ├── resnet/
│   └── comparison/
├── requirements.txt
└── README.md
```

`data/`, `outputs/`, and model weight files are intentionally excluded from Git using `.gitignore`. To reproduce the results, run the scripts locally and retain their generated artifacts for submission.

## 5. Training configuration

| Setting | YOLO11n | ResNet-18 |
|---|---|---|
| Initialization | `yolo11n.pt` pretrained detection model | ImageNet pretrained ResNet-18 |
| Objective | Detection on full images | Classification on ground-truth object crops |
| Classes | 11 | 9 |
| Max epochs | 20 | 20 |
| Input size | 640 | 224 × 224 |
| Batch size | 8 | 32 |
| Optimizer | Ultralytics auto selection | Adam, learning rate 1e-4 |
| Imbalance approach | Standard YOLO training | Inverse-frequency weighted cross-entropy |
| Checkpoint selection | Best validation fitness | Best validation accuracy |
| Device | NVIDIA RTX 4050 | NVIDIA RTX 4050 |

ResNet uses ImageNet normalization and random horizontal flips on training crops; validation and testing use deterministic preprocessing. Both models are fine-tuned, not trained from random initialization. Equal epochs do **not** equate the architectures, input representations, optimizer, effective update counts, or total training compute.

## 6. How to reproduce

Run commands at project root, in order:

```powershell
python src/utils/analyze_dataset.py
python src/utils/visualize_annotations.py
python src/yolo/train.py
python src/yolo/evaluate.py
python src/resnet/prepare_dataset.py
python src/resnet/train.py
python src/resnet/evaluate.py
python src/resnet/visualize_predictions.py
python src/yolo/evaluate_crops.py
python src/utils/compare_models.py
python src/utils/benchmark_models.py
python src/utils/make_final_figures.py
```

**Caution:** Running either training script again can start or resume training and may replace model files depending on checkpoint state. To generate final figures on an already completed project, run **only the last command** after copying `make_final_figures.py` to `src/utils/`.

## 7. Experimental results

### 7.1 YOLO11n full-image detection — independent test set (141 images; 1,251 objects)

| Metric | Result |
|---|---:|
| Precision | 52.97% |
| Recall | 51.68% |
| mAP@0.5 | 52.95% |
| mAP@0.5:0.95 | 26.78% |

Selected per-class test recall: `helmet` 92.19%, `vest` 89.89%, `no_helmet` 15.00%, `no_goggle` 9.09%, `no_gloves` 14.92%, `no_boots` 0.00%. Critical missing-PPE cases are frequently missed. Detection visualizations and confidence scores: `outputs/yolo/test_predictions/`.

### 7.2 ResNet-18 object classification — independent test set (950 crops)

| Metric | Result |
|---|---:|
| Accuracy | 89.37% (849 / 950) |
| Macro precision | 82.85% |
| Macro recall | 81.47% |
| Macro F1 | 81.90% |
| Best validation accuracy | 92.14% (epoch 15) |

`no_boots` is the weakest class (F1 0.46, 23 test crops). Saved evidence includes `outputs/resnet/confusion_matrix.png`, `classification_report.csv`, `classification_errors.png`, `test_predictions.csv`, and `prediction_confidence_samples.png`.

### 7.3 Identical-crop classification diagnostic (950 test crops)

| Metric | YOLO11n adapted as crop classifier | ResNet-18 |
|---|---:|---:|
| Accuracy | 20.42% | **89.37%** |
| Macro precision | 17.07% | **82.85%** |
| Macro recall | 19.17% | **81.47%** |
| Macro F1 | 13.19% | **81.90%** |

YOLO returned no PPE detection on **352/950 crops (37.05%)**, counted as incorrect in the diagnostic. Its highest-confidence detection, when present, was treated as the crop label. This is an *adaptation of the detector*, not a native YOLO classification evaluation. Consequently, these results should not be interpreted as showing that the native detector is inferior to ResNet for surveillance; they demonstrate that the fine-tuned ResNet is considerably better at the separate task of classifying provided crops.

### 7.4 Single-crop inference benchmark (RTX 4050, batch size 1, 100 crops)

| Metric | YOLO11n | ResNet-18 |
|---|---:|---:|
| Mean latency | 42.365 ms/crop | 10.035 ms/crop |
| Derived throughput | 23.604 crops/s | 99.656 crops/s |
| Model parameters | 2.607 million | 11.181 million |
| Measured peak allocated GPU memory | 65.981 MB | 139.567 MB |

The timing script used identical crop files, a ten-image warm-up, and CUDA synchronization. It includes in-memory image preprocessing and prediction calls, but excludes disk I/O and model loading. These results do **not** include any crop-localization pipeline for ResNet and do not give comparable end-to-end site-monitoring throughput. GPU memory peaks are approximate and were measured in a shared process, not isolated benchmark runs.

## 8. Interpretation and recommendation

**Recommend YOLO as the primary *full-image detection architecture* for an officer-assistive prototype**, because it identifies multiple objects, reports bounding boxes, and produces confidence-tagged detections without needing pre-existing crop coordinates. ResNet gives stronger nine-class classification of ground-truth crops, but cannot independently find or associate PPE items with particular workers. A possible hybrid—detection, crop classification, worker association, and alert triage—is future work, not a validated system.

**Do not deploy the current detector for automatic compliance decisions.** Very low recall for `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots` makes missed hazards likely. Priorities for improvement include richer missing-PPE annotations, class-balanced data acquisition, per-class threshold tuning using validation only, worker-to-PPE association, testing on independent real site imagery, calibrated alerts, and human review.

The dataset alone does not establish whether someone has entered a *restricted zone*: that requires camera calibration or zone definitions and a separate spatial-rule layer. Occluded or unobservable PPE should not automatically be interpreted as absent.

## 9. Evidence / submission files

- `outputs/comparison/dataset_distribution.csv`
- `outputs/yolo/ppe_training/results.png`, `weights/best.pt`
- `outputs/yolo/test_evaluation/` and `test_metrics.csv`
- `outputs/yolo/test_predictions/` (boxes, labels, confidence)
- `outputs/resnet/training_history.csv`, `best_resnet18.pth`
- `outputs/resnet/classification_report.csv`, `confusion_matrix.png`
- `outputs/resnet/classification_errors.csv`, `classification_errors.png`
- `outputs/resnet/test_predictions.csv`, `prediction_confidence_samples.png`
- `outputs/comparison/yolo_crop_predictions.csv`, `classification_comparison.csv`, `model_characteristics.csv`, `inference_benchmark.csv`
- `outputs/comparison/final_classification_comparison.png`, `final_inference_comparison.png` (generated by `make_final_figures.py`)

## 10. Reproducibility / limitations

- Split by original image, *then* crop. No test crops are used to fine-tune either model.
- A fixed seed was set in the ResNet script; GPU training and resumed runs may not be bitwise deterministic.
- Model checkpoints are chosen using validation data, never test data.
- Different objectives, batch sizes, optimizers, and input sizes are recorded rather than misrepresented as identical.
- Classification confidence is the ResNet softmax maximum; YOLO scores represent detection confidence. Neither is necessarily calibrated.
- `none` was excluded from the crop experiment because its intended semantics were not confirmed.

## 11. Dataset and library credit

Construction-PPE dataset and YOLO software are provided by Ultralytics. Check their respective dataset usage terms and software [AGPL-3.0/commercial licensing](https://www.ultralytics.com/license) before distribution or commercial deployment. PyTorch/Torchvision provide pretrained ResNet-18. Refer to official upstream documentation for model and data details.
