from pathlib import Path
import math
import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "undistorted"
OUTPUT_DIR = IMAGE_DIR / "contact_sheets"

COLUMNS = 5
ROWS = 4
THUMB_WIDTH = 320
THUMB_HEIGHT = 240
HEADER_HEIGHT = 32
GAP = 8

def make_thumbnail(image, filename):
    height, width = image.shape[:2]
    scale = min(THUMB_WIDTH / width, THUMB_HEIGHT / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = 255 * (image[:1, :1] * 0)
    canvas = cv2.copyMakeBorder(
        resized,
        (THUMB_HEIGHT - new_height) // 2,
        THUMB_HEIGHT - new_height - (THUMB_HEIGHT - new_height) // 2,
        (THUMB_WIDTH - new_width) // 2,
        THUMB_WIDTH - new_width - (THUMB_WIDTH - new_width) // 2,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245)
    )

    return canvas

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IMAGE_DIR.glob("*.jpg"))

    if len(files) != 129:
        raise RuntimeError(
            f"Expected 129 undistorted JPGs, found {len(files)}"
        )

    images = []
    dimensions = {}

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read: {path.name}")

        height, width = image.shape[:2]
        dimensions[(width, height)] = dimensions.get((width, height), 0) + 1

        thumbnail = make_thumbnail(image, path.name)
        images.append((path.name, thumbnail))

    per_sheet = COLUMNS * ROWS
    sheet_count = math.ceil(len(images) / per_sheet)

    for sheet_index in range(sheet_count):
        batch = images[
            sheet_index * per_sheet:(sheet_index + 1) * per_sheet
        ]

        cell_width = THUMB_WIDTH
        cell_height = THUMB_HEIGHT + HEADER_HEIGHT

        sheet_width = COLUMNS * cell_width + (COLUMNS + 1) * GAP
        sheet_height = ROWS * cell_height + (ROWS + 1) * GAP

        sheet = 255 * (images[0][1][:1, :1] * 0)
        sheet = cv2.cvtColor(
            cv2.UMat(sheet).get(),
            cv2.COLOR_BGR2GRAY
        )
        sheet = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
        sheet = cv2.resize(sheet, (sheet_width, sheet_height))
        sheet[:] = (235, 235, 235)

        for index, (filename, thumbnail) in enumerate(batch):
            row = index // COLUMNS
            column = index % COLUMNS

            x = GAP + column * (cell_width + GAP)
            y = GAP + row * (cell_height + GAP)

            cv2.putText(
                sheet,
                filename,
                (x + 5, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA
            )

            sheet[
                y + HEADER_HEIGHT:y + HEADER_HEIGHT + THUMB_HEIGHT,
                x:x + THUMB_WIDTH
            ] = thumbnail

        output_path = OUTPUT_DIR / f"contact_sheet_{sheet_index + 1:02d}.jpg"

        if not cv2.imwrite(str(output_path), sheet):
            raise RuntimeError(f"Could not write: {output_path}")

        print(f"Saved: {output_path}")

    print("\nImage dimensions:")
    for dimension, count in sorted(dimensions.items()):
        print(f"  {dimension[0]} x {dimension[1]}: {count}")

    print(f"\nContact sheets created: {sheet_count}")
    print(f"Images included: {len(images)}")

if __name__ == "__main__":
    main()