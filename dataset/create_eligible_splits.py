
from pathlib import Path
import csv
import json
import cv2

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "dataset" / "splits"
MASK_DIR = ROOT / "dataset" / "labels" / "masks"
REPORT = ROOT / "dataset" / "labels" / "reports" / "report_20261107_231617.csv"

# These five existing masks were visually inspected and accepted.
VERIFIED_EXISTING = {
    f"img_{i:03d}.jpg" for i in range(1, 6)
}

def read_split(path):
    return [
        line.strip().replace("\ufeff", "")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

with REPORT.open("r", encoding="utf-8-sig", newline="") as f:
    report = {row["image"]: row for row in csv.DictReader(f)}

eligible = {"train": [], "val": [], "test": []}
excluded = []

for split in eligible:
    original = read_split(SPLIT_DIR / f"{split}.txt")

    for image in original:
        row = report.get(image)
        reason = None

        if row is None:
            reason = "missing_from_labelling_report"
        else:
            status = row["status"]
            accepted = (
                status == "ok"
                or (
                    status == "skipped_existing"
                    and image in VERIFIED_EXISTING
                )
            )

            if not accepted:
                reason = f"status_{status}"

        mask_path = MASK_DIR / f"{Path(image).stem}.png"

        if reason is None and not mask_path.is_file():
            reason = "mask_file_missing"

        if reason is None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None or cv2.countNonZero(mask) == 0:
                reason = "mask_empty_or_unreadable"

        if reason:
            excluded.append({
                "image": image,
                "split": split,
                "reason": reason
            })
        else:
            eligible[split].append(image)

for split, images in eligible.items():
    output = SPLIT_DIR / f"{split}_eligible.txt"
    output.write_text(
        "\n".join(images) + "\n",
        encoding="utf-8"
    )
    print(f"{split}: {len(images)} eligible / {len(read_split(SPLIT_DIR / f'{split}.txt'))} original")

exclusion_path = SPLIT_DIR / "excluded_images.json"
exclusion_path.write_text(
    json.dumps(excluded, indent=2),
    encoding="utf-8"
)

print(f"\nExcluded images: {len(excluded)}")
print(f"Exclusion report: {exclusion_path}")
print("\nOriginal split files were not modified.")
