
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.unet import UNet


# Match training configuration exactly.
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 288
PROBABILITY_THRESHOLD = 0.5
NUM_THREADS = 4

IMAGES_DIR = ROOT / "dataset" / "undistorted"
MASKS_DIR = ROOT / "dataset" / "labels" / "masks"
TEST_LIST = ROOT / "dataset" / "splits" / "test_eligible.txt"
CHECKPOINT = ROOT / "models" / "outputs" / "best_unet.pt"
OUTPUT_DIR = ROOT / "evaluation" / "outputs"


def load_test_items():
    if not TEST_LIST.is_file():
        raise FileNotFoundError(TEST_LIST)

    items = [
        line.strip()
        for line in TEST_LIST.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if len(items) != 10:
        raise RuntimeError(
            f"Expected 10 eligible test images, found {len(items)}."
        )

    if len(items) != len(set(items)):
        raise RuntimeError("Duplicate filenames in test split.")

    return items


def load_sample(filename):
    image_path = IMAGES_DIR / filename
    mask_path = MASKS_DIR / (Path(filename).stem + ".png")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    if target is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    target = cv2.resize(
        target,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )

    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(
        image.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    target = target > 0

    return image, target


def calculate_mask_iou(prediction, target):
    intersection = np.logical_and(
        prediction, target
    ).sum()

    union = np.logical_or(
        prediction, target
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def calculate_ap(records, threshold):
    """
    Single-object-per-image mask AP.

    One prediction is produced per image.
    A prediction is a true positive when its mask IoU
    reaches the specified threshold.

    AP uses all-point interpolated precision.
    """
    ranked = sorted(
        records,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    true_positives = []
    false_positives = []

    for item in ranked:
        is_positive = (
            item["prediction_nonempty"]
            and item["iou"] >= threshold
        )

        true_positives.append(1 if is_positive else 0)
        false_positives.append(0 if is_positive else 1)

    tp = np.cumsum(true_positives, dtype=np.float64)
    fp = np.cumsum(false_positives, dtype=np.float64)

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / len(records)

    # Add boundary points and calculate precision envelope.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    changes = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum(
        (mrec[changes + 1] - mrec[changes])
        * mpre[changes + 1]
    )

    return float(ap)


def main():
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(42)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    items = load_test_items()

    device = torch.device("cpu")

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Final test evaluation")
    print(f"Test images: {len(items)}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation IoU: {checkpoint['best_val_iou']:.4f}")
    print(f"Device: {device}")
    print(f"Probability threshold: {PROBABILITY_THRESHOLD}")
    print("Test split is used only for final evaluation.")

    records = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for filename in items:
            image, target = load_sample(filename)
            image = image.to(device)

            logits = model(image)
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prediction = probabilities >= PROBABILITY_THRESHOLD

            target_pixels = target
            intersection = np.logical_and(
                prediction, target_pixels
            ).sum()

            tp = int(intersection)
            fp = int(
                np.logical_and(
                    prediction, np.logical_not(target_pixels)
                ).sum()
            )
            fn = int(
                np.logical_and(
                    np.logical_not(prediction), target_pixels
                ).sum()
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            iou = calculate_mask_iou(
                prediction,
                target_pixels,
            )

            if prediction.any():
                confidence = float(
                    probabilities[prediction].mean()
                )
            else:
                confidence = float(probabilities.max())

            record = {
                "image": filename,
                "iou": iou,
                "confidence": confidence,
                "prediction_nonempty": bool(prediction.any()),
                "target_pixels": int(target_pixels.sum()),
                "predicted_pixels": int(prediction.sum()),
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
            }

            records.append(record)

            print(
                f"{filename}: "
                f"IoU={iou:.4f}, "
                f"confidence={confidence:.4f}"
            )

    mean_iou = float(
        np.mean([item["iou"] for item in records])
    )

    pixel_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    pixel_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    ap_values = {}

    for threshold_percent in range(50, 100, 5):
        threshold = threshold_percent / 100.0
        ap = calculate_ap(records, threshold)
        ap_values[f"AP@{threshold:.2f}"] = ap

    map_50 = ap_values["AP@0.50"]
    map_50_95 = float(np.mean(list(ap_values.values())))

    results = {
        "evaluation": "held-out test split",
        "test_images": len(records),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_iou": float(checkpoint["best_val_iou"]),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "mean_mask_iou": mean_iou,
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "mAP_50": map_50,
        "mAP_50_95": map_50_95,
        "ap_by_iou_threshold": ap_values,
        "ap_protocol": (
            "Single-class, one prediction and one ground-truth "
            "object per image; predictions ranked by mean foreground "
            "probability; all-point interpolated precision."
        ),
        "per_image": records,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "test_metrics.json"
    csv_path = OUTPUT_DIR / "test_per_image.csv"

    json_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nFINAL TEST RESULTS")
    print(f"Mean mask IoU: {mean_iou:.4f}")
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_50_95:.4f}")
    print(f"Pixel precision: {pixel_precision:.4f}")
    print(f"Pixel recall: {pixel_recall:.4f}")

    print("\nAP by IoU threshold:")
    for name, value in ap_values.items():
        print(f"{name}: {value:.4f}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
