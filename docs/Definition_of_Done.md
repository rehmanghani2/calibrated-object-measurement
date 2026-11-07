# Definition of Done — Measure It

This checklist records the evidence available at the end of the assessment work. A requirement is marked complete only where there is supporting implementation or recorded output.

## Item 1 — Camera calibration

- [x] OpenCV calibration implementation created.
- [x] At least 20 calibration images detected.
- [x] Camera matrix and distortion coefficients saved.
- [x] Reprojection error reported.

**Evidence**
- Calibration script: `calibration/calibrate_camera.py`
- Parameters: `calibration/camera_calibration.npz`
- Report: `calibration/calibration_report.json`
- Per-image errors: `calibration/reprojection_errors.csv`
- 26 of 27 images detected; RMS reprojection error 4.3883 px.

**Limitation:** Calibration quality is provisional because the reprojection error is relatively high.

## Item 2 — Own labelled dataset

- [x] 129 original casing images inventoried.
- [x] Original train/validation/test split committed.
- [x] Images undistorted using saved calibration parameters.
- [x] Algorithmically generated masks quality-checked.
- [x] Eligible image-mask pairs checked for readability, non-empty masks, matching dimensions and split overlap.
- [x] Eligible split lists created.

**Evidence**
- Source images: `dataset/originals/`
- Preprocessing: `dataset/prepare_images.py`
- Annotation generation: `dataset/auto_label.py`
- Eligible split lists: `dataset/splits/train_eligible.txt`, `val_eligible.txt`, `test_eligible.txt`
- Verification: `dataset/verify_training_data.py`

**Dataset result:** 93 eligible pairs: 64 train, 19 validation and 10 test. 36 images were excluded after failure or review checks.

**Limitation:** The masks were algorithmically generated and manually quality-checked, not all individually hand-traced.

## Item 3 — Trained segmentation model

- [x] Custom segmentation architecture implemented.
- [x] Model trained using the training split.
- [x] Validation split used for checkpoint selection.
- [x] Test split kept out of training and checkpoint selection.
- [x] Final test metrics recorded.

**Evidence**
- Model: `models/unet.py`
- Training: `training/train.py`
- Checkpoint: `models/outputs/best_unet.pt`
- Training history: `models/outputs/history.csv`
- Evaluation: `evaluation/evaluate_test.py`

**Results:** Mean test IoU 0.9582; mAP@0.5 1.0000; mAP@0.5:0.95 0.9523; pixel precision 0.9687; pixel recall 0.9914.

## Item 4 — Image-to-millimetres pipeline

- [x] One-command inference implementation created.
- [x] Image undistortion included.
- [x] Segmentation model loaded and run.
- [x] Reference card detection and perspective rectification implemented.
- [x] Width, height and segmentation confidence output.
- [x] Missing or implausible reference geometry can cause rejection.

**Evidence**
- Pipeline: `inference/measure.py`
- Example output: `measurement/results/`
- Accuracy evaluation logs: `measurement/accuracy/run_logs/`

**Limitation:** The reference quadrilateral is estimated from the red pattern region, and the calibration reprojection error is relatively high.

## Item 5 — Measurement accuracy

- [ ] Ten or more successful system measurements documented.
- [x] Ruler ground truth documented.
- [x] Per-instance results recorded for successful measurements.
- [x] MAE and percentage error calculated for successful measurements.
- [x] Failed measurement retained in the evaluation record.

**Evidence**
- Evaluation script: `measurement/run_accuracy_evaluation.py`
- Per-instance table: `measurement/accuracy/per_instance_results.csv`
- Summary: `measurement/accuracy/summary.json`
- Report: `measurement/accuracy/accuracy_report.md`
- Test image list: `measurement/accuracy/test_images_used.txt`

**Result:** 9 of 10 original test images measured successfully. `img_068.jpg` was rejected because the detected card aspect ratio was 1.962.

**Status:** Incomplete. The successful results are repeated photographs of one physical casing, not ten independent physical objects.

## Item 6 — Undistortion comparison

- [ ] Raw-image and undistorted-image measurements compared on the same instances.
- [ ] Error results reported side by side.
- [ ] Difference explained using measured results.

**Status:** Not completed.

## Item 7 — Speed or accuracy target

- [ ] One target formally defined.
- [ ] Measurement procedure executed.
- [ ] Result and target attainment reported.

**Status:** Not completed as a separately evaluated objective.

## Item 8 — Breaking condition

- [ ] One condition selected and varied.
- [ ] Measurement error quantified as the condition worsened.
- [ ] Reason for selecting the condition documented.

**Status:** Not completed. This was not started because Item 5 remained incomplete.

## Item 9 — Refusal behaviour

- [x] A reference-geometry rejection was observed and recorded.
- [ ] A complete refusal threshold documented.
- [ ] Refusal cost quantified across evaluation images.

**Evidence:** `measurement/accuracy/run_logs/img_068_console.txt`

**Status:** Partially completed. The observed rejection is not a complete refusal-threshold evaluation.

## Item 10 — Decision record

- [x] Main implementation choices and limitations documented in `Plan.md`.
- [ ] Formal comparison of chosen and rejected approaches, including quantified cost.

**Status:** Partially completed.

## Submission checklist

- [ ] All three documents reviewed and committed under `docs/`.
- [ ] Root README reviewed for setup, execution instructions, hardware, dataset summary and model SHA-256.
- [ ] Latest changes pushed to `dev-test01`.
- [ ] Pull request opened from `dev-test01` into `main`.
- [ ] Loom recording completed, no longer than five minutes.
- [ ] Pull request and Loom links submitted through the required response channel.

## Final status

The calibration, dataset preparation, model training and core inference pipeline have been implemented. Segmentation evaluation has been completed. Measurement accuracy remains incomplete, and Items 6–8 have not been completed. Item 9 is only partially evidenced.

This checklist intentionally distinguishes implemented functionality from requirements that have not been demonstrated.