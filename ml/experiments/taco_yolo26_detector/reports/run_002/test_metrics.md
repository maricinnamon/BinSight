# Run 002 (epoch 20 best.pt, 24 epochs trained) — detection metrics

Object-detection metrics. There is no single 'accuracy' figure here on
purpose: this model localises *and* classifies, and a classification-style
accuracy would describe neither job.

## Overall

| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.5219 | 0.2703 | 0.3562 | 0.2947 | 0.2546 | 0.2241 |
| test | 0.3158 | 0.2470 | 0.2772 | 0.2105 | 0.1637 | 0.1453 |

## Per class — val

| ID | Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 0.5480 | 0.2957 | 0.3841 | 0.2971 | 0.2155 |
| 1 | bottle | 0.6619 | 0.2448 | 0.3575 | 0.3721 | 0.2581 |
| 2 | bottle_cap | 0.4513 | 0.3056 | 0.3644 | 0.3094 | 0.2381 |
| 3 | can | 0.3990 | 0.5143 | 0.4494 | 0.4887 | 0.3764 |
| 4 | carton | 0.4024 | 0.2450 | 0.3045 | 0.2823 | 0.2369 |
| 5 | cigarette | 0.8220 | 0.0125 | 0.0246 | 0.0243 | 0.0200 |
| 6 | cup | 0.3957 | 0.4615 | 0.4261 | 0.4559 | 0.3586 |
| 7 | straw | 0.4948 | 0.0833 | 0.1426 | 0.1277 | 0.0892 |

## Per class — test

| ID | Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 0.3985 | 0.3304 | 0.3613 | 0.2870 | 0.1899 |
| 1 | bottle | 0.4248 | 0.3462 | 0.3815 | 0.3281 | 0.2246 |
| 2 | bottle_cap | 0.4299 | 0.3611 | 0.3925 | 0.3316 | 0.1945 |
| 3 | can | 0.3934 | 0.4684 | 0.4276 | 0.3549 | 0.2797 |
| 4 | carton | 0.2397 | 0.2727 | 0.2552 | 0.1698 | 0.1413 |
| 5 | cigarette | 0.5054 | 0.0120 | 0.0235 | 0.0576 | 0.0214 |
| 6 | cup | 0.1351 | 0.1852 | 0.1562 | 0.1374 | 0.0986 |
| 7 | straw | 0.0000 | 0.0000 | 0.0000 | 0.0175 | 0.0123 |
