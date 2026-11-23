"""Genesis ARC V2.3 reproducible research experiment runner."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

_V22_PATH = Path(__file__).resolve().parents[1] / "v2.2"
if str(_V22_PATH) not in sys.path:
    sys.path.insert(0, str(_V22_PATH))

from factorization_experiment import run_factorization
from active_experiment import run_active, summarize
from online_demo import run_online_demo

NOISES = (0.0, 0.05, 0.20)

def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    factor_rows = run_factorization(args.seeds)
    active_rows = run_active(args.seeds, args.noises)
    active_summary = summarize(active_rows)
    online = run_online_demo()
    write_csv(out / "factorization.csv", factor_rows)
    write_csv(out / "active_trials.csv", active_rows)
    write_csv(out / "active_summary.csv", active_summary)
    (out / "online_demo.json").write_text(json.dumps(online, indent=2), encoding="utf-8")
    metadata = {
        "seeds": args.seeds, "noises": list(args.noises), "state_count": 8,
        "latent_operators_per_factorization_trial": 4, "pairwise_relations_per_trial": 12,
        "active_candidate_count": 4, "active_discovery_queries": 6,
        "active_validation_inputs": 4, "active_verification_repeats_per_input": 3,
        "active_confidence_threshold": 0.995,
        "open_set_null_model": "uniform output per query; illustrative, not universal novelty detector",
        "factorization_assumption": "complete directed pairwise composites C_ij = P_j o inverse(P_i)",
        "caveat": "finite permutation-map experiment; no general-intelligence claim"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("anchored exact:", sum(r["anchored_exact_recovery"] for r in factor_rows), "/", len(factor_rows))
    print("canonical consistent:", sum(r["unanchored_canonical_consistent"] for r in factor_rows), "/", len(factor_rows))
    print("gauge invariant:", sum(r["gauge_invariant"] for r in factor_rows), "/", len(factor_rows))
    for row in active_summary:
        print(row)
    print("online demo:", online)

def selftest():
    factor = run_factorization(4)
    assert all(r["anchored_exact_recovery"] for r in factor)
    assert all(r["unanchored_canonical_consistent"] for r in factor)
    assert all(r["gauge_invariant"] for r in factor)
    assert all(not r["unanchored_original_unique"] for r in factor)
    online = run_online_demo()
    assert online["candidate_accepted"] and online["old_record_still_decodes_before_migration"]
    assert online["second_old_record_still_decodes_before_migration"]
    assert online["atomic_migration_succeeded"] and online["audit"]["roundtrip_rate"] == 1.0
    from active_experiment import candidate_family
    from open_set import identify_open_set
    candidates, unknown, _ = candidate_family(55)
    known = identify_open_set(candidates, candidates[2], noise=0.0, strategy="active")
    novel = identify_open_set(candidates, unknown, noise=0.0, strategy="active")
    assert known.decision == "identified" and known.correct
    assert novel.decision == "rejected_unknown"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=256)
    p.add_argument("--noises", type=float, nargs="+", default=list(NOISES))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.seeds < 1 or any(not 0 <= x < 1 for x in args.noises):
        raise ValueError("invalid experiment parameters")
    selftest()
    run(args)

if __name__ == "__main__":
    main()
