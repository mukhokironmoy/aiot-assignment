# Comparative Analysis of YOLO11n and ResNet-18 for Construction PPE Monitoring

## 1. Abstract

This project investigates two transfer-learning approaches for personal protective equipment (PPE) assessment in construction-site imagery. A YOLO11n object detector was trained to identify and localize eleven object categories, while a ResNet-18 classifier was fine-tuned on nine classes of ground-truth object crops. Both were trained for 20 epochs on an NVIDIA RTX 4050 laptop GPU. On 141 held-out full test images, YOLO achieved precision 0.5297, recall 0.5168, mAP@0.5 0.5295, and mAP@0.5:0.95 0.2678. On 950 held-out PPE crops, ResNet achieved 0.8937 accuracy and 0.8190 macro F1. An adapted YOLO crop-classification procedure reached 0.2042 accuracy, reflecting domain/input mismatch and missing detections rather than an equivalent native-task comparison. YOLO is recommended as the primary framework for localization-driven officer assistance, but its low missing-PPE recall precludes trustworthy automatic compliance decisions in its present form.

## 2. Data and experimental methodology

The Construction-PPE dataset comprises 1,416 images, distributed into 1,132 training, 143 validation, and 141 test images. The corresponding object annotation totals are 9,098, 1,172, and 1,251. The eleven annotation labels are `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots`. Bounding boxes were read from YOLO's normalized class-and-coordinate text format.

For ResNet, nine PPE-specific classes were retained, and annotated boxes were cropped into 6,677 training, 852 validation, and 950 test examples. `Person` and `none` were excluded from the classification study. Original image-level splits were retained, so test images and derived test crops were not included in the model-training split. A basic structural annotation check found zero invalid entries, but semantic annotation correctness was not independently audited.

YOLO11n was initialized from `yolo11n.pt` and trained on full images at image size 640, batch size 8, and 20 epochs. ResNet-18 was ImageNet-initialized, its output head replaced with nine classes, and all layers fine-tuned using image size 224×224, batch size 32, Adam learning rate 1e-4, and inverse-frequency weighted cross-entropy. The best ResNet checkpoint was selected on validation accuracy at epoch 15 (92.14%). These are architecture-appropriate settings, not identical training conditions despite equal epoch counts.

## 3. Results

### 3.1 Native YOLO detection on the 141-image test set

| Metric | Value |
|---|---:|
| Precision | 52.97% |
| Recall | 51.68% |
| mAP@0.5 | 52.95% |
| mAP@0.5:0.95 | 26.78% |

**Violation-class recall:** `no_helmet` 15.00%, `no_goggle` 9.09%, `no_gloves` 14.92%, `no_boots` 0.00%. By comparison, recall was 92.19% for `helmet` and 89.89% for `vest`. These findings indicate that the main safety-critical errors are often missed violations, rather than missing detections of ordinary visible PPE.

### 3.2 Native ResNet classification on the 950-crop test set

| Metric | Value |
|---|---:|
| Accuracy | 89.37% (849/950 correct) |
| Macro precision | 82.85% |
| Macro recall | 81.47% |
| Macro F1 | 81.90% |

Performance was strongest for `vest` (F1 0.97) and `helmet` (F1 0.96), and weakest for `no_boots` (F1 0.46). There were 101 misclassified crops. A confusion matrix and confidence-labelled predictions were generated to inspect errors.

### 3.3 Controlled-input diagnostic: classifying the same 950 crops

| Metric | Adapted YOLO11n | ResNet-18 |
|---|---:|---:|
| Accuracy | 20.42% | 89.37% |
| Macro precision | 17.07% | 82.85% |
| Macro recall | 19.17% | 81.47% |
| Macro F1 | 13.19% | 81.90% |

For YOLO, the highest-confidence detected PPE box was assigned as the crop label and no detection counted as an error. There were 352 no-detection outcomes (37.05% of 950 crops). ResNet's advantage here is specific to the supplied-crop classification task; the modified YOLO procedure is not a native classification model or a test of full-image localization performance.

### 3.4 Computational benchmark on 100 test crops

| Measurement | YOLO11n | ResNet-18 |
|---|---:|---:|
| Mean single-crop latency | 42.365 ms | 10.035 ms |
| Derived throughput | 23.604 crops/s | 99.656 crops/s |
| Parameter count | 2.607M | 11.181M |
| Reported peak GPU allocated memory | 65.981 MB | 139.567 MB |

The benchmark used batch size 1, same crop files, GPU warm-up and synchronization, and excluded file loading/model initialization. It is **not** a comparison of complete monitoring pipelines: standalone ResNet cannot locate crops. Memory was not measured in isolated processes and therefore should be treated as approximate.

## 4. Architectural comparison

YOLO is a one-stage object detector: it processes full images and returns multiple class labels, bounding boxes, and confidence scores. ResNet-18 is a residual convolutional classifier: it receives an already-defined crop and outputs class scores (converted to probabilities using softmax), without spatial localization. YOLO should be evaluated natively using localization-sensitive precision/recall and mAP metrics; ResNet should be evaluated using classification accuracy, per-class precision/recall/F1, and a confusion matrix. Confidence values are model scores, not guaranteed calibrated probabilities of correctness.

## 5. Practical recommendation

**Recommend YOLO as the primary architecture for full-frame construction-site triage**, because multiple people and PPE items need to be located before actionable alerts can be displayed to safety officers. The current fine-tuned YOLO checkpoint should **not** be used for autonomous safety decisions due to very low missing-PPE recall. ResNet is appropriate as a complementary classifier only when meaningful object crops are provided by an upstream localization system. An integrated detector–classifier design may be evaluated as future work, but was not implemented or validated by the current experiments.

For future improvement, collect and label more rare violation examples, especially `no_boots`; inspect class definitions and ambiguous labels; tune operating thresholds on the validation split; explicitly associate equipment with the correct worker; evaluate on independent site images and difficult lighting/occlusion conditions; and keep a human reviewer in the alert loop. Restricted-area access requires defined camera zones or spatial rules, which neither model implements by itself.

## 6. Conclusion

The assignment demonstrates why good object classification does not automatically imply good site-wide PPE monitoring. ResNet provides strong accuracy on ground-truth PPE crops, while YOLO delivers required localization capability but misses many safety-critical violations. The appropriate deployment decision is therefore task-driven: use detection to support human inspection, treat low-confidence or incomplete observations cautiously, and improve missing-PPE recall before operational use.

## 7. Figures and supplementary artifacts

- YOLO full-image detections: `outputs/yolo/test_predictions/`
- YOLO test statistics: `outputs/yolo/test_metrics.csv` and `test_evaluation/`
- ResNet confusion matrix: `outputs/resnet/confusion_matrix.png`
- ResNet correct/incorrect confidence display: `outputs/resnet/prediction_confidence_samples.png`
- ResNet classification report: `outputs/resnet/classification_report.csv`
- Same-crop model predictions: `outputs/comparison/yolo_crop_predictions.csv` and `outputs/resnet/test_predictions.csv`
- Final side-by-side charts: `outputs/comparison/final_classification_comparison.png`, `final_inference_comparison.png`, `final_error_examples.png`

All numerical results in this report reflect the provided experiment terminal outputs and CSV excerpts. Images and individual stored predictions remain in the local project and were not independently inspected for this report.
