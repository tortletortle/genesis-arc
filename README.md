# Genesis ARC

> Private research repository. Each version keeps its own source, experiments, and explicit validation limits.

## Research direction

Study skill acquisition, compositional generalization, memory, and continual adaptation in controlled synthetic environments. These research prototypes are not a claim of general intelligence or a complete digital-life ecosystem.

## Current research branches

- **V1.9** — continual skill retrieval and adaptive compute baseline.
- **V2.0** — evidence-based skill consolidation and frozen recall probes. Draft PR #1.
- **V2.1** — lossless raw/program/delta memory compression. Draft PR #2.
- **V2.2** — self-organizing dictionary induction from observed skill maps, pair look-ahead, held-out composition tests, and hidden-primitive limitation tests. Draft PR #3.

V2.0–V2.2 are staged as incremental research branches. V2.2 is not merged into `main`, and it is not integrated with the V1.5.1 evolutionary-population/ecosystem line.

## V2.2 quick start

```bash
cd versions/v2.2
pip install -r requirements.txt
python -c 'import experiment; experiment.selftest()'
python -m unittest discover -s tests -v
python experiment.py --seeds 16 --noises .05 .2 --out results
python analyze.py results
```

See [V2.2 report](versions/v2.2/V2.2_REPORT.md) for the experiment design, results, and limitations.

## Engineering rules

1. Keep versioned source and baselines immutable once used as an experimental reference.
2. Preserve raw results, test outputs, and configuration metadata.
3. Separate measured facts, inferences, and unresolved questions.
4. Do not label a feature complete without reproducible tests.
5. No open-source license has been selected yet.
