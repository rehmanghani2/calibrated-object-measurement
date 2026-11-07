@'
# Calibrated Object Measurement

Computer-vision assessment project for measuring mobile-phone casing dimensions in millimetres using camera calibration, image segmentation, and a known-size reference object.

## Project status

Planning and initial dataset organisation.

## Hardware and environment

- Camera: Samsung Galaxy J5 (2016)
- Development CPU: Intel Core i5-8365U
- RAM: 16 GB
- Graphics: Intel UHD Graphics 620
- Operating system: Windows (development environment)
- Training compute: To be confirmed and documented.

## Dataset

- Original casing photographs: 129 JPG images.
- Original image location: `dataset/originals/`.
- Training split: 90 images.
- Validation split: 26 images.
- Test split: 13 images.
- Split lists: `dataset/splits/`.
- Calibration checkerboard photographs: 27 images are available in the supplied archive and will be organised under `calibration/images/`.

The casing photographs contain a mobile-phone casing and a playing card used as an in-image measurement reference.

The reported playing-card dimensions are 56.5 mm × 87.0 mm. These dimensions must be physically verified with callipers before being used for final measurement.

## Planned technical approach

1. Calibrate the camera using OpenCV and checkerboard images.
2. Undistort the original photographs.
3. Create and review binary segmentation masks.
4. Train a U-Net model with a ResNet-34 encoder.
5. Detect the playing-card reference and estimate casing dimensions.
6. Evaluate segmentation, measurement accuracy, runtime, and failure handling.

The final model and measurement results have not yet been produced.

## Repository structure

```text
calibrated-object-measurement/
├── calibration/
│   └── images/
├── dataset/
│   ├── originals/
│   ├── images/
│   ├── labels/
│   └── splits/
├── docs/
├── inference/
├── measurement/
│   └── results/
├── models/
└── README.md