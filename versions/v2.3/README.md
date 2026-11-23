# Genesis ARC V2.3 — Latent Discovery & Online Evolution

**Status:** research prototype; draft PR, not merged.

V2.3 tests three bounded capabilities on finite 8-state permutation maps:

1. **Latent-factor identifiability.** Under the explicit complete relation model `C[i,j] = P[j] o inverse(P[i])`, one known anchor recovers all original operators. Without an anchor, only a relative basis is identifiable because a shared right-composition leaves every observation unchanged.
2. **Active discrimination.** Mutual-information query selection is compared with fixed-order and random queries. Four input states are reserved for independent validation; a uniform-output null hypothesis permits candidate-set rejection.
3. **Online dictionary evolution.** Immutable dictionary snapshots, record version pinning, MDL candidate addition, budget pruning, and all-or-nothing in-memory migration preserve old decoders.

## Measured results

- 256/256 anchored factorization trials recovered all operators exactly.
- 256/256 unanchored canonical models explained the observed pairwise relations; the original operators remained gauge-ambiguous with at least `8! = 40,320` equivalent tuples.
- At 20% observation noise, active correct-decision rate was 96.88% for in-set maps and 91.02% for out-of-set maps, versus 89.45% / 75.39% for fixed queries and 85.16% / 83.59% for random queries.
- At 20% noise, active false acceptance of unknown maps was 2.73%; fixed was 3.13%, random 2.34%. Active does not dominate every error metric.
- Online demo: abstract storage cost 168 to 136 bits; two old records remained decodable across the update and migrated with 100% round-trip success.

## Reproduction

Download and extract [the complete source package](genesis_arc_v23_source.zip) into this directory, then run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -c 'import experiment; experiment.selftest()'
python experiment.py --seeds 256 --noises 0 .05 .2 --out results_full
python analyze.py results_full
```

The archive contains source, tests, full research report, experiment metadata, per-seed factorization records, and aggregate summaries. The 4,608-row active-trial CSV can be regenerated with the experiment command.

## Important limits

- The pairwise relation structure is supplied by the synthetic experiment; V2.3 does not yet discover that structure from arbitrary unlabelled skills.
- No anchor means no unique recovery of original operator identities.
- The open-set null assumes uniform independent outputs; it is not a calibrated universal novelty detector.
- Online versioning and migration are in-memory only, not durable or crash-safe.
- Bit counts use V2.2's abstract model, not actual serialized bytes, RAM, latency, or energy.
- This branch is not integrated with the V1.5.1 evolutionary ecosystem and makes no general-intelligence claim.

See `V2.3_REPORT.md` inside the source package for methods, Wilson intervals, detailed results, and the V2.4 research agenda.
