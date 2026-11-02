
from pathlib import Path
import time

import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/comparison"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WARMUP = 10
RUNS = 100


def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark(name, predict_fn, images, parameters):
    # Warm up the GPU before measuring
    for image in images[:WARMUP]:
        predict_fn(image)

    synchronize()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []

    for image in images:
        synchronize()
        start = time.perf_counter()

        predict_fn(image)

        synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    average_ms = sum(times) / len(times)

    memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if DEVICE == "cuda"
        else 0
    )

    return {
        "model": name,
        "avg_inference_ms": average_ms,
        "images_per_second": 1000 / average_ms,
        "parameters_millions": parameters / 1e6,
        "peak_gpu_memory_mb": memory_mb,
    }


def main():
    print("Device:", DEVICE)

    # Load 100 identical test crops
    records = pd.read_csv(
        ROOT / "outputs/resnet/test_predictions.csv"
    )

    paths = [
        ROOT / path
        for path in records["image"].head(RUNS)
    ]

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    print("Benchmark images:", len(images))

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo = YOLO(str(
        ROOT / "outputs/yolo/ppe_training/weights/best.pt"
    ))

    def predict_yolo(image):
        yolo.predict(
            source=image,
            imgsz=640,
            conf=0.01,
            device=DEVICE,
            verbose=False,
        )

    yolo_params = sum(
        p.numel() for p in yolo.model.parameters()
    )

    yolo_results = benchmark(
        "YOLO11n",
        predict_yolo,
        images,
        yolo_params,
    )

    # --------------------------------------------------------
    # RESNET
    # --------------------------------------------------------

    checkpoint = torch.load(
        ROOT / "outputs/resnet/best_resnet18.pth",
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    resnet = models.resnet18(weights=None)
    resnet.fc = torch.nn.Linear(
        resnet.fc.in_features,
        len(class_names),
    )

    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet = resnet.to(DEVICE)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    def predict_resnet(image):
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            output = resnet(tensor)
            torch.softmax(output, dim=1)

    resnet_params = sum(
        p.numel() for p in resnet.parameters()
    )

    resnet_results = benchmark(
        "ResNet18",
        predict_resnet,
        images,
        resnet_params,
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame([
        yolo_results,
        resnet_results,
    ])

    OUTPUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT / "inference_benchmark.csv",
        index=False,
    )

    print("\n--- INFERENCE BENCHMARK ---")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
