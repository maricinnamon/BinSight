# Object sizes: Run 001 vs recovered dataset

The question this answers: did recovery add *different* objects, or more of the same size profile? A shifted profile changes what Run 002 can be expected to learn; an unchanged one means the gain is purely volume.

| Class | Small 001 | Small 002 | Small % 001 | Small % 002 | Δ pp |
|---|---:|---:|---:|---:|---:|
| `bag_wrapper` | 91 | 168 | 21.2% | 22.0% | +0.9 |
| `bottle` | 14 | 56 | 9.2% | 16.0% | +6.8 |
| `bottle_cap` | 50 | 143 | 38.5% | 60.1% | +21.6 |
| `can` | 8 | 33 | 6.2% | 14.0% | +7.8 |
| `carton` | 8 | 26 | 9.4% | 11.8% | +2.4 |
| `cigarette` | 167 | 518 | 89.8% | 96.5% | +6.7 |
| `cup` | 6 | 23 | 7.2% | 13.1% | +5.8 |
| `straw` | 31 | 58 | 32.6% | 37.9% | +5.3 |

Overall small-object share moved from **29.1%** to **38.4%** — a shift of **+9.3 pp**. The largest per-class move is **`bottle_cap` at +21.6 pp**.

**This is not a flat rescale — the recovered dataset is harder.** The images Flickr refused were not a random sample of TACO: they carried proportionally more tiny objects, so recovery raised the small-object share rather than leaving it alone.

Two consequences for reading Run 002:

1. Run 002's mAP is **not directly comparable** to Run 001's as a measure of the model. The test set changed and got harder, so a flat or slightly lower mAP would still be consistent with a better detector.
2. The honest comparison is per-class and size-bucketed, not a single headline number. `evaluate_run.py` already reports per class; the size buckets in `full_object_size_distribution.md` are what to read it against.

Run 001 remains frozen precisely so this comparison stays available — its metrics were measured on its own splits and are not retroactively invalidated, only differently scoped.

