# Measurement Accuracy Evaluation

- Successful measurements: 9 / 10
- Failed measurements: 1
- Evaluation set: committed eligible test split
- Ground truth: 78.0 mm width × 160.2 mm height
- Ground truth method: ruler measurement of the physical casing
- Object count: one physical casing, photographed repeatedly

## Per-instance results

| Image | GT width | Measured width | Width error | GT height | Measured height | Height error | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| img_062.jpg | 78.00 | 79.27 | 1.27 | 160.20 | 166.61 | 6.41 | 0.9590 |
| img_033.jpg | 78.00 | 89.64 | 11.64 | 160.20 | 155.89 | 4.31 | 0.9754 |
| img_030.jpg | 78.00 | 104.73 | 26.73 | 160.20 | 157.72 | 2.48 | 0.9729 |
| img_094.jpg | 78.00 | 58.44 | 19.56 | 160.20 | 145.85 | 14.35 | 0.9804 |
| img_127.jpg | 78.00 | 100.69 | 22.69 | 160.20 | 178.60 | 18.40 | 0.9642 |
| img_092.jpg | 78.00 | 52.73 | 25.27 | 160.20 | 146.17 | 14.03 | 0.9658 |
| img_035.jpg | 78.00 | 98.62 | 20.62 | 160.20 | 172.77 | 12.57 | 0.9606 |
| img_024.jpg | 78.00 | 84.31 | 6.31 | 160.20 | 174.58 | 14.38 | 0.9610 |
| img_089.jpg | 78.00 | 74.01 | 3.99 | 160.20 | 155.58 | 4.62 | 0.9729 |

## Aggregate errors

| Metric | Width | Height |
|---|---:|---:|
| MAE (mm) | 15.3422 | 10.1722 |
| Median absolute error (mm) | 19.5600 | 12.5700 |
| Worst absolute error (mm) | 26.7300 | 18.4000 |
| Worst-case image | img_030.jpg | img_127.jpg |
| Mean absolute percentage error (%) | 19.6695 | 6.3497 |

## Evaluation limitations

- The ten images show the same physical casing; they are not ten independent objects.
- The same ruler-measured dimensions are used as ground truth for each photograph.
- Results depend on the accuracy of the ruler measurements and the assumption that the casing and reference card lie in the same plane.
- Segmentation confidence is not a calibrated probability of measurement correctness.
- Failed measurements are listed in summary.json and are not silently counted as successful measurements.

## Failed instances

- img_068.jpg: Inference returned exit code 1
