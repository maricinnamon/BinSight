# Run 003 decision — resolution ablation

**Decision: TRAIN**

Run 003 justified: tiny_recall_below, small_recall_below, tiny_small_fn_share_above. Tiny recall 0.0949 (threshold 0.20), small recall 0.2125 (threshold 0.30), tiny+small share of false negatives 86.5% (threshold 40%). Run 002 misses gate A on: map50_95, recall, tiny_recall.

## Gate A — is Run 002 already good enough to skip?

Skip only if **all three** pass.

| Check | Value | Threshold | Passes |
|---|---:|---:|---|
| map50_95 | 0.1453 | >= 0.3 | no |
| recall | 0.2470 | >= 0.35 | no |
| tiny_recall | 0.0949 | >= 0.2 | no |

## Gate B — is resolution the dominant failure mode?

Train only if **at least one** triggers.

| Check | Value | Threshold | Triggered |
|---|---:|---:|---|
| tiny_recall_below | 0.0949 | 0.2 | yes |
| small_recall_below | 0.2125 | 0.3 | yes |
| tiny_small_fn_share_above | 0.8645 | 0.4 | yes |

## Competing explanations

Checked so a low score is not blamed on resolution by default:

- Class confusion is **26.5%** of all detections (35 of 132). Higher resolution does not fix a model that finds the object and names it wrong.
- False negatives by size bucket: {'tiny': 218, 'medium': 32, 'small': 50, 'large': 10}.
- Label corruption was ruled out before training: the dataset validator reported 0 problems over 2671 boxes.

Thresholds were fixed before Run 002 finished, in the brief for this step, so the decision could not be tuned to the result.

