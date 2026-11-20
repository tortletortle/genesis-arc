# Genesis ARC V1.9
Experimental continual learning for synthetic discrete worlds. See [research report](V1.9_REPORT.md) for design, results and limitations.

```bash
pip install -r requirements.txt
python -c 'import run; run.selftest()'
python run.py --seeds 8 --rounds 6 --out results
python run.py --seeds 8 --rounds 6 --noise .2 --offset 10000 --out robustness
python analyze.py
```

The baseline V1.8 code is preserved under `baseline_v18/`. Results are synthetic experiments, not evidence of general intelligence. No license has been selected by the owner.
