# yolo26n_run_001 vs yolo26n_run_002_full_taco_640

> **The two runs were scored on different test splits.** Recovery enlarged the pool from 509 to 1082 images and raised the COCO-small share from 29.1% to 38.4%, so Run 002's test set is harder as well as different. These deltas measure model-plus-data, not the model alone; an equal number on a harder split is a real gain.

## Overall — test split

| Metric | run_001 | run_002 | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| precision | 0.2696 | 0.3158 | +0.0462 | +17.1% |
| recall | 0.1853 | 0.2470 | +0.0617 | +33.3% |
| f1 | 0.2197 | 0.2772 | +0.0575 | +26.2% |
| map50 | 0.2007 | 0.2105 | +0.0098 | +4.9% |
| map75 | 0.1696 | 0.1637 | -0.0060 | -3.5% |
| map50_95 | 0.1586 | 0.1453 | -0.0133 | -8.4% |

## Overall — validation split

| Metric | run_001 | run_002 | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| precision | 0.4483 | 0.5219 | +0.0736 | +16.4% |
| recall | 0.2619 | 0.2703 | +0.0084 | +3.2% |
| f1 | 0.3306 | 0.3562 | +0.0256 | +7.7% |
| map50 | 0.2861 | 0.2947 | +0.0086 | +3.0% |
| map75 | 0.2357 | 0.2546 | +0.0190 | +8.1% |
| map50_95 | 0.2120 | 0.2241 | +0.0121 | +5.7% |

## Recall by object size — test split

Identical bucket definitions to Run 001: tiny < 1% of frame area, small < 5%, medium < 20%, large above.

| Bucket | run_001 | run_002 | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| tiny | 0.0511 | 0.0949 | +0.0438 | +85.7% |
| small | 0.1739 | 0.2125 | +0.0386 | +22.2% |
| medium | 0.3077 | 0.2727 | -0.0350 | -11.4% |
| large | 0.5000 | 0.2353 | -0.2647 | -52.9% |

## Per class — test AP50-95

| Class | run_001 | run_002 | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| `bag_wrapper` | 0.1156 | 0.1899 | +0.0744 | +64.3% |
| `bottle` | 0.4194 | 0.2246 | -0.1948 | -46.5% |
| `bottle_cap` | 0.2288 | 0.1945 | -0.0343 | -15.0% |
| `can` | 0.1031 | 0.2797 | +0.1766 | +171.3% |
| `carton` | 0.0167 | 0.1413 | +0.1246 | +743.9% |
| `cigarette` | 0.0015 | 0.0214 | +0.0199 | +1360.7% |
| `cup` | 0.3761 | 0.0986 | -0.2775 | -73.8% |
| `straw` | 0.0078 | 0.0123 | +0.0045 | +57.6% |

## Per class — test recall

| Class | run_001 | run_002 | Δ abs | Δ rel |
|---|---:|---:|---:|---:|
| `bag_wrapper` | 0.1194 | 0.3304 | +0.2110 | +176.7% |
| `bottle` | 0.3333 | 0.3462 | +0.0128 | +3.8% |
| `bottle_cap` | 0.3799 | 0.3611 | -0.0188 | -5.0% |
| `can` | 0.1500 | 0.4684 | +0.3184 | +212.3% |
| `carton` | 0.0000 | 0.2727 | +0.2727 | — |
| `cigarette` | 0.0000 | 0.0120 | +0.0120 | — |
| `cup` | 0.5000 | 0.1852 | -0.3148 | -63.0% |
| `straw` | 0.0000 | 0.0000 | +0.0000 | — |

## Dataset and training cost

| | run_001 | run_002 | Δ abs |
|---|---:|---:|---:|
| dataset images | — | 1082 | — |
| dataset annotations | — | 2671 | — |
| training images | — | 748 | — |
| training annotations | — | 1861 | — |
| epochs_completed | 64 | 24 | -40.0000 |
| best_epoch | 44 | 20 | -24.0000 |
| training_hours | 1.5900 | 2.1840 | +0.5940 |

No classification accuracy appears here. These are detection runs and a classification-style accuracy would describe neither localisation nor multi-object frames.

