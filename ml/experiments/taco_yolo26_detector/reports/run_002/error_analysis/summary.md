# Error analysis — Run 002 (epoch 20 best.pt) (test split)

Matching: greedy by descending confidence, **IoU >= 0.5**, prediction confidence >= 0.25.

| | |
|---|---:|
| Ground-truth objects | 405 |
| Predictions | 132 |
| True positives | 60 |
| False positives | 37 |
| False negatives | 310 |
| Class confusions (right box, wrong class) | 35 |
| Poor localisation (0.3 <= IoU < 0.5, right class) | 2 |

## False negatives by class

| Class | Missed |
|---|---:|
| cigarette | 83 |
| bag_wrapper | 82 |
| bottle | 36 |
| bottle_cap | 26 |
| carton | 23 |
| straw | 22 |
| can | 20 |
| cup | 18 |

## False positives by predicted class

| Predicted class | Count |
|---|---:|
| bag_wrapper | 10 |
| carton | 7 |
| cup | 6 |
| can | 6 |
| bottle | 5 |
| bottle_cap | 2 |
| straw | 1 |

## Confusion pairs

| Ground truth | Predicted as | Count |
|---|---|---:|
| bottle | can | 6 |
| bag_wrapper | bottle | 4 |
| bag_wrapper | cup | 3 |
| cup | can | 3 |
| bag_wrapper | carton | 3 |
| carton | cup | 3 |
| can | bag_wrapper | 2 |
| cup | bag_wrapper | 2 |
| bag_wrapper | can | 1 |
| carton | bag_wrapper | 1 |
| carton | bottle | 1 |
| can | bottle_cap | 1 |

## Misses by object size

| Bucket | Ground truth | Detected | Recall |
|---|---:|---:|---:|
| tiny | 253 | 24 | 0.095 |
| small | 80 | 17 | 0.212 |
| medium | 55 | 15 | 0.273 |
| large | 17 | 4 | 0.235 |

Small-object recall is the number that matters for BinSight: the eventual
camera sees objects at a distance, and TACO's median object is under 1% of
the frame.

