# Supervisor mechanics smoke test — **PASS**

Purpose is trainer mechanics **only**. The metrics from a 4-epoch run mean
nothing and are deliberately not quoted.

## What was proven

| Check | Result |
|---|---|
| child A trains and exits cleanly at its epoch budget | True |
| checkpoint written and verified between children | True |
| child B starts and resumes (`resume=True`) | True |
| same logical run directory throughout | `runs/smoke_mechanics` — one directory, no `run2` |
| `results.csv` continues rather than restarting | epochs [1, 2, 3, 4] |
| no duplicate epoch rows | True |
| optimizer state survives the recycle | True |
| EMA survives the recycle | True |
| class count preserved | nc=8 |
| unexpected kills | 0 |

## Children

| # | Mode | Epochs | Exit | Classification |
|---:|---|---|---:|---|
| 1 | start | 1–2 | 0 | planned_recycle |
| 2 | resume | 3–4 | 0 | final_child |

## Memory across the recycle

101 samples logged to `reports/memory_log_smoke.csv`.
Swap ranged 2017–2406 MB; peak 2406 MB,
below the 3584 MB early-recycle threshold, so no memory-triggered recycle fired.

The child's own before/after cleanup snapshot is recorded in each
`reports/child_NNN.log`. On MPS the meaningful reclamation happens when the
process exits — `gc.collect()` and `torch.mps.empty_cache()` are called first so
the logged numbers are honest, not because they substitute for process exit.

## One artefact worth knowing

Ultralytics' `time` column in `results.csv` restarts at zero in each new child
(158, 314, 159, 313 s here). Wall-clock duration for the whole run is therefore
the **sum of segments**, not the last value. The epoch *index* continues
correctly, which is what matters for resume.
