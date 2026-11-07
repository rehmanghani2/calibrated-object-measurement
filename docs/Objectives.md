# Project Objectives --- Calibrated Object Measurement

**Project:** `calibrated-object-measurement`\
**Status:** Draft objectives; targets are engineering targets, not
measured results.

## MO-1 --- Measurement accuracy

On the untouched test set, compare automated casing width and height
against physical calliper measurements. The proposed target is:

-   Mean absolute percentage error (MAPE) no greater than **5.0%**
    across evaluated width and height measurements.
-   Report width and height separately as well as the combined result.
-   Report per-instance absolute error in millimetres and percentage
    error, MAE, median absolute error and worst-case error.

For a measurement (p_i) and calliper ground truth (g_i):

Absolute error (mm) = |prediction - ground truth|

Absolute percentage error (%) =
    (absolute error / ground truth) × 100

MAE (mm) =
    sum of absolute errors / number of measurements

MAPE (%) =
    sum of absolute percentage errors / number of measurements

The 5.0% value is a proposed acceptance target, not a result and not a
value stated by the assessment. Ground-truth measurements must be taken
with callipers and recorded with units, tool and date. Do not substitute
the preliminary user-reported casing dimensions for per-instance ground
truth.

## MO-2 --- End-to-end inference performance

On the documented development machine, measure end-to-end CPU inference
time for five runs on the same representative image, including image
loading, undistortion, segmentation, reference detection and
measurement, but excluding one-time model loading. The proposed target
is:

-   Median end-to-end processing time no greater than **5.0 seconds per
    image**.
-   Report all five timings, median, minimum and maximum, along with
    CPU, RAM, OS, image resolution and software versions.

The 5.0-second value is a proposed engineering target, not a measured
result or an assessment-specified threshold. If the final implementation
cannot meet it, report the measured result honestly and explain the
bottleneck.

## Supporting model-quality measures

Report on the held-out test set:

-   Binary mask IoU.
-   Precision and recall.
-   mAP@0.5 and mAP@0.5:0.95.

Because the selected model is semantic binary segmentation, document
exactly how predicted masks are converted to instances/boxes for mAP.
Use the same evaluation protocol consistently for all reported results.
Do not present an undefined mAP number.

## Evaluation integrity

-   Freeze and commit the train, validation and test file lists before
    training.
-   Use validation data for model selection and threshold selection
    only.
-   Keep the test split isolated until the final evaluation.
-   Report sample counts and exclusions; do not silently omit failed
    cases.
-   Report abstention coverage and refusal rate alongside accuracy so
    that accuracy cannot be improved merely by refusing difficult
    images.
-   All reported numbers must be generated from saved machine-readable
    outputs or documented measurement records. Clearly label targets,
    measured results and illustrative examples.
