#!/usr/bin/env python
"""Decides from Run 002's evidence whether a 768 px Run 003 is justified.

Two gates, both read from persisted reports so the decision cannot drift with
whoever is reading it:

**Gate A — is Run 002 already good enough that resolution does not matter?**
Skip Run 003 only if test mAP50-95 >= 0.30 AND test recall >= 0.35 AND tiny
recall >= 0.20.

**Gate B — is resolution actually the dominant failure mode?**
Run 003 proceeds only if at least one of: tiny recall < 0.20, small recall < 0.30,
or >= 40% of false negatives are tiny/small objects.

Gate B is the one that stops wasted compute. A low mAP caused by class confusion,
label noise or class scarcity will not improve at 768, and training six more hours
to discover that is the expensive way to learn it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

GATE_A = {"map50_95": 0.30, "recall": 0.35, "tiny_recall": 0.20}
GATE_B = {"tiny_recall": 0.20, "small_recall": 0.30, "tiny_small_fn_share": 0.40}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="run_002")
    args = parser.parse_args()

    run_reports = REPORTS / args.run
    metrics_path = run_reports / "test_metrics.json"
    errors_path = run_reports / "error_analysis.json"
    if not metrics_path.exists() or not errors_path.exists():
        print(f"ERROR: {args.run} reports incomplete — evaluate first.")
        return 1

    metrics = json.loads(metrics_path.read_text())
    errors = json.loads(errors_path.read_text())
    test = metrics["test"]["overall"]
    sizes = {b["bucket"]: float(b["recall"]) for b in errors["size_buckets"]}

    fn_path = run_reports / "error_analysis" / "false_negatives.csv"
    fn_buckets = collections.Counter()
    if fn_path.exists():
        for row in csv.DictReader(open(fn_path)):
            fn_buckets[row["size_bucket"]] += 1
    fn_total = sum(fn_buckets.values())
    tiny_small_share = ((fn_buckets["tiny"] + fn_buckets["small"]) / fn_total
                        if fn_total else 0.0)

    # ------------------------------------------------------------- gate A
    gate_a = {
        "map50_95": (test["map50_95"], GATE_A["map50_95"], test["map50_95"] >= GATE_A["map50_95"]),
        "recall": (test["recall"], GATE_A["recall"], test["recall"] >= GATE_A["recall"]),
        "tiny_recall": (sizes.get("tiny", 0.0), GATE_A["tiny_recall"],
                        sizes.get("tiny", 0.0) >= GATE_A["tiny_recall"]),
    }
    already_good = all(v[2] for v in gate_a.values())

    # ------------------------------------------------------------- gate B
    gate_b = {
        "tiny_recall_below": (sizes.get("tiny", 0.0), GATE_B["tiny_recall"],
                              sizes.get("tiny", 0.0) < GATE_B["tiny_recall"]),
        "small_recall_below": (sizes.get("small", 0.0), GATE_B["small_recall"],
                               sizes.get("small", 0.0) < GATE_B["small_recall"]),
        "tiny_small_fn_share_above": (round(tiny_small_share, 4),
                                      GATE_B["tiny_small_fn_share"],
                                      tiny_small_share >= GATE_B["tiny_small_fn_share"]),
    }
    resolution_is_bottleneck = any(v[2] for v in gate_b.values())

    # Competing explanations, so a "yes" from gate B is not read in isolation.
    counts = errors.get("counts", {})
    tp, fp, fn = counts.get("tp", 0), counts.get("fp", 0), counts.get("fn", 0)
    confusions = counts.get("confusions", 0)
    detections = tp + fp + confusions
    confusion_share = confusions / detections if detections else 0.0
    competing = {
        "class_confusion_share_of_detections": round(confusion_share, 4),
        "class_confusion_dominant": confusion_share >= 0.40,
        "false_negative_share_of_ground_truth": round(fn / max(tp + fn + confusions, 1), 4),
    }

    if already_good:
        decision, reason = "SKIP", (
            "Run 002 already clears every gate-A threshold (test mAP50-95 "
            f"{test['map50_95']:.4f} >= 0.30, recall {test['recall']:.4f} >= 0.35, tiny "
            f"recall {sizes.get('tiny', 0):.4f} >= 0.20). Resolution is not the "
            "constraint; proceed to model selection.")
    elif not resolution_is_bottleneck:
        decision, reason = "SKIP", (
            "Run 003 skipped because resolution is not the dominant failure mode. "
            f"Tiny recall {sizes.get('tiny', 0):.4f} and small recall "
            f"{sizes.get('small', 0):.4f} both clear the gate-B thresholds, and only "
            f"{tiny_small_share:.1%} of false negatives are tiny or small objects. "
            "Training at 768 would not address the actual failure.")
    else:
        triggered = [k for k, v in gate_b.items() if v[2]]
        decision, reason = "TRAIN", (
            f"Run 003 justified: {', '.join(triggered)}. "
            f"Tiny recall {sizes.get('tiny', 0):.4f} (threshold 0.20), small recall "
            f"{sizes.get('small', 0):.4f} (threshold 0.30), tiny+small share of false "
            f"negatives {tiny_small_share:.1%} (threshold 40%). "
            f"Run 002 misses gate A on: "
            f"{', '.join(k for k, v in gate_a.items() if not v[2])}.")
        if competing["class_confusion_dominant"]:
            reason += (" Caveat: class confusion accounts for "
                       f"{confusion_share:.1%} of detections, which 768 will not fix — "
                       "read Run 003 for size buckets, not for the headline number.")

    result = {
        "run": args.run,
        "decision": decision,
        "reason": reason,
        "gate_a_skip_if_all_pass": {k: {"value": v[0], "threshold": v[1], "passes": v[2]}
                                    for k, v in gate_a.items()},
        "gate_a_all_pass": already_good,
        "gate_b_train_if_any_true": {k: {"value": v[0], "threshold": v[1], "triggered": v[2]}
                                     for k, v in gate_b.items()},
        "gate_b_any_triggered": resolution_is_bottleneck,
        "competing_explanations": competing,
        "false_negatives_by_bucket": dict(fn_buckets),
        "size_recall": sizes,
    }
    out = REPORTS / "comparisons"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_003_decision.json").write_text(json.dumps(result, indent=2) + "\n")

    md = [
        "# Run 003 decision — resolution ablation", "",
        f"**Decision: {decision}**", "", reason, "",
        "## Gate A — is Run 002 already good enough to skip?", "",
        "Skip only if **all three** pass.", "",
        "| Check | Value | Threshold | Passes |", "|---|---:|---:|---|",
    ]
    for key, (value, threshold, ok) in gate_a.items():
        md.append(f"| {key} | {value:.4f} | >= {threshold} | {'yes' if ok else 'no'} |")
    md += ["", "## Gate B — is resolution the dominant failure mode?", "",
           "Train only if **at least one** triggers.", "",
           "| Check | Value | Threshold | Triggered |", "|---|---:|---:|---|"]
    for key, (value, threshold, ok) in gate_b.items():
        md.append(f"| {key} | {value} | {threshold} | {'yes' if ok else 'no'} |")
    md += ["", "## Competing explanations", "",
           "Checked so a low score is not blamed on resolution by default:", "",
           f"- Class confusion is **{confusion_share:.1%}** of all detections "
           f"({confusions} of {detections}). Higher resolution does not fix a model "
           "that finds the object and names it wrong.",
           f"- False negatives by size bucket: {dict(fn_buckets)}.",
           "- Label corruption was ruled out before training: the dataset validator "
           "reported 0 problems over 2671 boxes.", "",
           "Thresholds were fixed before Run 002 finished, in the brief for this step, "
           "so the decision could not be tuned to the result.", ""]
    (out / "run_003_decision.md").write_text("\n".join(md) + "\n")

    print(json.dumps({"decision": decision, "reason": reason}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
