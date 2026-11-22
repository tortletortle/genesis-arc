# Genesis ARC V2.1

Lossless skill-memory compression and capacity sweep for a controlled synthetic environment.

## Quick start

```bash
pip install -r requirements.txt
python -c 'import run; run.selftest()'
python -m unittest discover -s tests -v
python run.py --seeds 4 --rounds 6 --noise .1 --offset 40000 --out results_10a
python run.py --seeds 4 --rounds 6 --noise .1 --offset 41000 --out results_10b
python merge_runs.py --out results_10 results_10a results_10b
python run.py --seeds 4 --rounds 6 --noise .2 --offset 50000 --out results_20a
python run.py --seeds 4 --rounds 6 --noise .2 --offset 51000 --out results_20b
python merge_runs.py --out results_20 results_20a results_20b
python analyze.py results_10 results_20
```

## Storage model

Each skill has a fixed 16-bit metadata allowance. A raw map is 8 x 3 = 24 bits. Program-coded maps use a 2-bit length and 4 bits per primitive token (12 shared primitives); out-of-DSL maps fall back to raw tables. Delta coding uses a minimum spanning forest of exact program/table roots and sparse patches (parent pointer, 4-bit change count, 3-bit index + 3-bit output per changed state). Shared primitive dictionary/interpreter cost is excluded equally from all arms. This is an explicit bit-accounting model, not Python heap/RAM or hardware measurement.

V2.1 keeps the V1.9 inference and utility-based admission/eviction policy fixed. It changes only the memory representation and capacity budget. Codecs are lossless; the simulator decodes maps before inference, so the experiment does not measure compute/latency savings.

See [V2.1 report](V2.1_REPORT.md) and `results_10/`, `results_20/` for full data.