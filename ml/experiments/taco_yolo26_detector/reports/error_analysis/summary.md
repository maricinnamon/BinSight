# Error analysis — Run 001 (test split)

Matching: greedy by descending confidence, **IoU >= 0.5**, prediction confidence >= 0.25.

| | |
|---|---:|
| Ground-truth objects | 211 |
| Predictions | 58 |
| True positives | 24 |
| False positives | 18 |
| False negatives | 171 |
| Class confusions (right box, wrong class) | 16 |
| Poor localisation (0.3 <= IoU < 0.5, right class) | 2 |

## False negatives by class

| Class | Missed |
|---|---:|
| bag_wrapper | 62 |
| cigarette | 27 |
| straw | 27 |
| bottle | 15 |
| can | 14 |
| bottle_cap | 12 |
| carton | 10 |
| cup | 4 |

## False positives by predicted class

| Predicted class | Count |
|---|---:|
| bag_wrapper | 5 |
| can | 3 |
| carton | 3 |
| cup | 2 |
| bottle | 2 |
| bottle_cap | 2 |
| cigarette | 1 |

## Confusion pairs

| Ground truth | Predicted as | Count |
|---|---|---:|
| carton | can | 3 |
| cup | can | 2 |
| bottle | bag_wrapper | 2 |
| can | bag_wrapper | 2 |
| cup | bottle_cap | 2 |
| cigarette | bag_wrapper | 1 |
| bottle | can | 1 |
| can | bottle | 1 |
| can | carton | 1 |
| bottle_cap | can | 1 |

## Misses by object size

| Bucket | Ground truth | Detected | Recall |
|---|---:|---:|---:|
| tiny | 137 | 7 | 0.051 |
| small | 46 | 8 | 0.174 |
| medium | 26 | 8 | 0.308 |
| large | 2 | 1 | 0.500 |

Small-object recall is the number that matters for BinSight: the eventual
camera sees objects at a distance, and TACO's median object is under 1% of
the frame.

