from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "undistorted"
MASK_DIR = ROOT / "dataset" / "labels" / "masks"
SPLIT_DIR = ROOT / "dataset" / "splits"

splits = ("train", "val", "test")
all_names = set()
failed = []

for split in splits:
    list_path = SPLIT_DIR / f"{split}_eligible.txt"
    names = [
        line.strip().lstrip("\ufeff")
        for line in list_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    print(f"\n{split}: {len(names)} eligible entries")

    for name in names:
        if name in all_names:
            failed.append(f"Duplicate across splits: {name}")
        all_names.add(name)

        image_path = IMAGE_DIR / name
        mask_path = MASK_DIR / f"{Path(name).stem}.png"

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            failed.append(f"Unreadable or missing image: {name}")
        if mask is None:
            failed.append(f"Unreadable or missing mask: {mask_path.name}")
        elif not (mask > 0).any():
            failed.append(f"Empty mask: {mask_path.name}")
        elif image is not None and image.shape[:2] != mask.shape[:2]:
            failed.append(f"Image/mask dimensions differ: {name}")

print(f"\nTotal unique eligible images: {len(all_names)}")

if failed:
    print("\nDATA CHECK FAILED:")
    for item in failed:
        print("-", item)
    raise SystemExit(1)

print("\nPASS: all eligible images and masks are readable, non-empty,")
print("dimension-matched, and assigned to only one split.")