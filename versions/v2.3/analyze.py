"""Statistical summaries for V2.3 raw seed-level experiment outputs."""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

def wilson(successes: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return float(center - half), float(center + half)

def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def summarize_active(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row["noise"]), row["truth_type"], row["strategy"])].append(row)
    output = []
    for (noise, truth_type, strategy), group in sorted(groups.items()):
        n = len(group)
        if truth_type == "in_set":
            correct = sum(r["decision"] == "identified" and r["correct"] == "True" for r in group)
            false_reject = sum(r["decision"] == "rejected_unknown" for r in group)
            false_accept = 0
            accepted = sum(r["decision"] == "identified" for r in group)
            accepted_correct = sum(r["decision"] == "identified" and r["correct"] == "True" for r in group)
        else:
            correct = sum(r["decision"] == "rejected_unknown" for r in group)
            false_reject = 0
            false_accept = sum(r["decision"] == "identified" for r in group)
            accepted = sum(r["decision"] == "identified" for r in group)
            accepted_correct = 0
        uncertain = sum(r["decision"] == "uncertain" for r in group)
        lo, hi = wilson(correct, n)
        fa_lo, fa_hi = wilson(false_accept, n)
        fr_lo, fr_hi = wilson(false_reject, n)
        output.append({
            "noise": noise, "truth_type": truth_type, "strategy": strategy, "n": n,
            "correct_decision_rate": correct / n, "correct_ci95_low": lo, "correct_ci95_high": hi,
            "false_accept_rate": false_accept / n, "false_accept_ci95_low": fa_lo,
            "false_accept_ci95_high": fa_hi, "false_reject_rate": false_reject / n,
            "false_reject_ci95_low": fr_lo, "false_reject_ci95_high": fr_hi,
            "uncertain_rate": uncertain / n, "accepted_rate": accepted / n,
            "accepted_correct_rate": accepted_correct / accepted if accepted else float("nan"),
            "mean_queries": float(np.mean([int(r["queries"]) for r in group])),
            "median_queries": float(np.median([int(r["queries"]) for r in group]))})
    return output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    folder = Path(p.parse_args().results_dir)
    active = load_csv(folder / "active_trials.csv")
    factor = load_csv(folder / "factorization.csv")
    summary = summarize_active(active)
    with (folder / "summary_detailed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    exact = sum(r["anchored_exact_recovery"] == "True" for r in factor)
    gauge = sum(r["gauge_invariant"] == "True" for r in factor)
    canonical = sum(r["unanchored_canonical_consistent"] == "True" for r in factor)
    fs = [{"n": len(factor), "anchored_exact_recovery": exact / len(factor),
           "unanchored_canonical_consistency": canonical / len(factor),
           "gauge_invariance": gauge / len(factor), "gauge_class_size": 40320}]
    with (folder / "factorization_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fs[0]))
        writer.writeheader(); writer.writerows(fs)
    for row in summary: print(row)
    print("Factorization:", fs[0])

if __name__ == "__main__":
    main()
