# Experimental design: Run 001 → Run 002

## The question Run 002 answers

**Was Run 001 limited by its model or by its data?**

Run 001 trained on 509 images — everything that survived a Flickr download in
which 898 of 1500 URLs returned HTTP 429. That is not a dataset choice, it is an
accident of rate limiting, and it makes Run 001's numbers unattributable: a weak
per-class result could mean YOLO26n cannot learn the class, or simply that the
class had 83 examples.

Run 002 removes the accident. Everything else is held fixed so that the answer
means something.

## Controlled variable

| | Run 001 | Run 002 (planned) |
|---|---|---|
| **Dataset** | 509 images / 1290 annotations | **1082 images / 2671 annotations** |
| Image source | Flickr, partial | Zenodo archive + Flickr, recovered |
| Classes | 8 | 8, identical IDs |
| Class mapping | `class_mapping.yaml` | byte-identical copy |
| `imgsz` | 640 | 640 |
| `model` init | `yolo26n.pt` | `yolo26n.pt` |
| `epochs` / `patience` | 100 / 20 | 100 / 20 |
| `batch` / `workers` | 8 / 4 | 8 / 4 |
| `seed` / `deterministic` | 42 / true | 42 / true |
| Optimiser, LR, loss gains | Ultralytics defaults | Ultralytics defaults |
| Augmentation | Ultralytics defaults | Ultralytics defaults |
| Device | `mps` | `mps` |
| Split rule | image-level, seed 42, 70/15/15 | same, plus content-hash grouping |

Exactly one row differs. Every training parameter was copied out of
`runs/yolo26n_run_001/args.yaml` — the settings Ultralytics resolved and actually
used — rather than re-derived from the config that requested them.

The splits gained one guard: Run 002 groups images by SHA-256 before assignment,
so byte-identical pictures cannot land on opposite sides of the train/test
boundary. It changed nothing in practice (all 1082 hashes are unique), which is
the point of checking.

## What Run 002 cannot answer

Three limits, stated before the run rather than after the result:

**1. The comparison is not a clean A/B on the same test set.** Run 002's test
split is drawn from a larger pool, so it is a different 166 images. Recovery also
made it harder: the small-object share rose from 29.1% to 38.4%, because the
images Flickr refused were not a random sample. A flat headline mAP would still
be consistent with a real improvement. Read per class and per size bucket —
`run_001_vs_full_object_sizes.md` is the reference.

**2. It is still not the complete TACO dataset.** 200 of 1500 image records
remain unresolved; the Zenodo archive predates the current annotation revision
and the remaining Flickr URLs are gone or rate-limited. Run 002 uses a
*maximally recovered official subset* (86.7%), not TACO. Anything trained on it
should say so.

**3. It does not test resolution.** Run 001's own config named 768 as the first
thing to try if small objects were the failure mode, and the size analysis
confirms they are. That remains untested on purpose: changing the dataset and
the input size in the same run would make neither attributable. 768 is Run 003.

## An annotation-quality observation

Rendering the cigarette crops (`reports/samples/run_002_dataset/tiny_objects/`)
shows something the numbers do not: several images contain visible cigarette
butts that carry no annotation, sitting beside one that does. TACO's ground truth
for this class is incomplete.

This has a direct consequence — a correct detection on an unlabelled butt is
scored as a false positive — so cigarette precision has a ceiling below 1.0 that
no amount of training reaches. This is an observation from looking at the
samples, not a measured rate; quantifying it would require re-annotating, which
is out of scope. It is recorded here so a low cigarette precision in Run 002 is
not misread as purely a model failure.

## Success criteria, fixed in advance

Set now so the result cannot be reinterpreted to fit whatever comes out:

- **Primary:** per-class AP on the Run 002 test split for the classes that
  roughly doubled — `cigarette` (186→537), `bag_wrapper` (430→762),
  `bottle` (152→349). If these rise materially, the bottleneck was data.
- **Secondary:** small-bucket AP against `full_object_size_distribution.md`. If
  small-object AP stays near zero while medium and large improve, the bottleneck
  is resolution and Run 003 at 768 is justified.
- **Negative result is a result.** If per-class AP is flat after 2.07× the data,
  that is evidence the limit is the architecture or the input size — and it is
  reported as such, not buried.

## Freeze policy

Run 001 stays exactly as it is: `runs/yolo26n_run_001/` is not resumed,
retrained, moved or overwritten, and `best.pt` still hashes to
`304aefbd0014b6e4e436e16d4c0b5d84ada6f46bb4d15c5d94a4494fc1039a69`.
`configs/dataset.yaml`, `data/prepared/` and `data/splits/` are its inputs and
are equally frozen — which is why `data/partial_run_001/` holds symlinks rather
than moved files. Run 002 writes only to `_run_002` paths.

**No Run 002 metrics exist. Run 002 has not been trained.**
