# Historical: the never-executed Colab notebook

This directory holds BinSight's **first** attempt at the TrashNet classifier: a
self-contained Colab notebook, written and statically validated but **never
run**. It was superseded by `../scripts/`, which is what actually produced the
trained model and every metric in `../reports/`.

Kept for project history. Nothing here is authoritative.

| File | What it is |
|---|---|
| `train_binsight_classifier.ipynb` | The notebook. Ships with no stored outputs, because it was never executed. |
| `requirements.notebook.txt` | Its intended environment (TensorFlow 2.19 / Colab Python 3.12) — **different** from the environment that trained the model (TF 2.13.1, Python 3.9.7, see `../environment.yml`). |
| `labels.template.txt` | Unfilled label template. Its order uses `general_waste` where the trained model emits `trash`. |
| `model_contract.template.json` | Unfilled contract template, `verified_by_execution: false`. |

**The authoritative files are `../labels.txt` and `../model_contract.json`**,
both generated from the real export and verified against the exported model.

The notebook targeted EfficientNet-Lite0 via KerasHub on a Colab runtime; the
executed pipeline used MobileNetV3Small on local CPU. Do not read the notebook
as a description of what was trained.
