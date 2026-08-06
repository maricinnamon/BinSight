# Run 001 — detection metrics

Object-detection metrics. There is no single 'accuracy' figure here on
purpose: this model localises *and* classifies, and a classification-style
accuracy would describe neither job.

## Overall

| Split | Precision | Recall | F1 | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| val | 0.4483 | 0.2619 | 0.3306 | 0.2861 | 0.2357 | 0.2120 |
| test | 0.2696 | 0.1853 | 0.2197 | 0.2007 | 0.1696 | 0.1586 |

## Per class — val

| ID | Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 0.4488 | 0.2656 | 0.3337 | 0.2499 | 0.1941 |
| 1 | bottle | 0.7938 | 0.3478 | 0.4837 | 0.4465 | 0.3407 |
| 2 | bottle_cap | 0.6025 | 0.5833 | 0.5928 | 0.5797 | 0.3998 |
| 3 | can | 0.6825 | 0.5238 | 0.5927 | 0.6277 | 0.5439 |
| 4 | carton | 0.4507 | 0.2143 | 0.2905 | 0.1845 | 0.1275 |
| 5 | cigarette | 0.0000 | 0.0000 | 0.0000 | 0.0400 | 0.0179 |
| 6 | cup | 0.2267 | 0.0769 | 0.1149 | 0.0465 | 0.0328 |
| 7 | straw | 0.3811 | 0.0833 | 0.1368 | 0.1141 | 0.0394 |

## Per class — test

| ID | Class | Precision | Recall | F1 | AP50 | AP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | bag_wrapper | 0.3968 | 0.1194 | 0.1836 | 0.1613 | 0.1156 |
| 1 | bottle | 0.5618 | 0.3333 | 0.4184 | 0.4572 | 0.4194 |
| 2 | bottle_cap | 0.4740 | 0.3799 | 0.4218 | 0.3264 | 0.2288 |
| 3 | can | 0.2374 | 0.1500 | 0.1838 | 0.1344 | 0.1031 |
| 4 | carton | 0.0000 | 0.0000 | 0.0000 | 0.0189 | 0.0167 |
| 5 | cigarette | 0.0000 | 0.0000 | 0.0000 | 0.0039 | 0.0015 |
| 6 | cup | 0.4870 | 0.5000 | 0.4934 | 0.4837 | 0.3761 |
| 7 | straw | 0.0000 | 0.0000 | 0.0000 | 0.0198 | 0.0078 |
