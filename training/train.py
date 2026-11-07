
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Reproducible configuration
# ---------------------------------------------------------

SEED = 42
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 288
BATCH_SIZE = 2
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
NUM_THREADS = 4

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "models" / "outputs"

IMAGES_DIR = ROOT / "dataset" / "undistorted"
MASKS_DIR = ROOT / "dataset" / "labels" / "masks"
SPLITS_DIR = ROOT / "dataset" / "splits"


TRAIN_LIST = SPLITS_DIR / "train_eligible.txt"
VAL_LIST = SPLITS_DIR / "val_eligible.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class CasingDataset(Dataset):
    def __init__(self, split_file: Path, augment: bool = False):
        self.augment = augment
        self.items = [
            line.strip()
            for line in split_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

        if not self.items:
            raise RuntimeError(f"No images listed in {split_file}")

        for filename in self.items:
            image_path = IMAGES_DIR / filename
            mask_path = MASKS_DIR / (Path(filename).stem + ".png")

            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        filename = self.items[index]

        image_path = IMAGES_DIR / filename
        mask_path = MASKS_DIR / (Path(filename).stem + ".png")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask > 0).astype(np.float32)

        image = image.astype(np.float32) / 255.0

        # Convert HWC image to CHW tensor.
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask[None, :, :].copy())

        # Horizontal flip only; apply the same transform to image and mask.
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        return image, mask


# ---------------------------------------------------------
# Load the U-Net defined in models/unet.py
# ---------------------------------------------------------


def create_model():
    import sys
    import inspect
    from pathlib import Path
    import torch.nn as nn

    sys.path.insert(0, str(ROOT))

    from models.unet import UNet

    if not inspect.isclass(UNet) or not issubclass(UNet, nn.Module):
        raise TypeError("UNet must be a PyTorch nn.Module class.")

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != 1_942_577:
        raise RuntimeError(
            f"Unexpected U-Net parameter count: {parameter_count:,}. "
            "Expected 1,942,577. Check models/unet.py."
        )

    print(f"Model class: {model.__class__.__name__}")
    print(f"Trainable parameters: {parameter_count:,}")

    return model


# ---------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------

def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def calculate_iou(logits, targets, threshold=0.5):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5

    intersection = (predictions & targets).sum(dim=(1, 2, 3))
    union = (predictions | targets).sum(dim=(1, 2, 3))

    iou = (intersection + 1e-7) / (union + 1e-7)

    return iou.mean().item()


# ---------------------------------------------------------
# Training and validation
# ---------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, training):
    model.train(training)

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            if logits.shape != masks.shape:
                raise RuntimeError(
                    f"Output/mask shape mismatch: "
                    f"{logits.shape} vs {masks.shape}"
                )

            loss = criterion(logits, masks) + dice_loss(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_iou += calculate_iou(logits.detach(), masks) * batch_size
            total_samples += batch_size

    return (
        total_loss / total_samples,
        total_iou / total_samples,
    )


def main():
    set_seed(SEED)
    torch.set_num_threads(NUM_THREADS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_file in (TRAIN_LIST, VAL_LIST):
        if not split_file.is_file():
            raise FileNotFoundError(split_file)

    train_dataset = CasingDataset(TRAIN_LIST, augment=True)
    val_dataset = CasingDataset(VAL_LIST, augment=False)

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("Test split: NOT LOADED")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = create_model().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_iou = -1.0
    epochs_without_improvement = 0
    history = []

    checkpoint_path = OUTPUT_DIR / "best_unet.pt"
    history_path = OUTPUT_DIR / "history.csv"
    config_path = OUTPUT_DIR / "training_config.json"

    config = {
        "seed": SEED,
        "architecture": "Custom U-Net",
        "input_width": IMAGE_WIDTH,
        "input_height": IMAGE_HEIGHT,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss + Dice loss",
        "device": str(device),
        "train_split": str(TRAIN_LIST.relative_to(ROOT)),
        "validation_split": str(VAL_LIST.relative_to(ROOT)),
        "test_split_loaded": False,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
    }

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_iou = run_epoch(
            model, train_loader, optimizer, criterion, device, True
        )

        val_loss, val_iou = run_epoch(
            model, val_loader, optimizer, criterion, device, False
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train IoU: {train_iou:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val IoU: {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "config": config,
                },
                checkpoint_path,
            )

            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_iou",
                    "val_loss",
                    "val_iou",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation IoU did not improve.")
            break

    print("\nTraining finished.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print("The test split has not been used.")


if __name__ == "__main__":
    main()
