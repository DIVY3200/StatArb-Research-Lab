# v3 experiment results (committed copies)

The experiment runner writes to `reports/experiments/`, which is not version-controlled. The folder below is a
**byte-identical copy** of the v3 development run, committed so the evidence behind
[`../DEVELOPMENT_REPORT.md`](../DEVELOPMENT_REPORT.md) is available from a clone.

| Folder | Stage | Notes |
|---|---|---|
| `20261024T183817Z_development_v3` | Development, 2015-01-01 to 2026-09-30 (evaluation 2017-01-03 to 2026-09-30) | Not frozen yet. The holdout (2026-10-01 to 2028-09-30) has not been evaluated. |

Contents as for v2 (see [`../../results/README.md`](../../results/README.md)), plus:

* `pooled_excess_returns.csv`: the pooled daily excess returns behind the Sharpe ratios and bootstrap intervals;
* `assumption_checks.csv`: the same signals re-priced with cash interest off, and under the v2 execution and cost
  assumptions;
* `windows_*.csv`: for every walk-forward window, Engle-Granger in both directions and the Johansen statistics
  (`suite_*` columns), plus the gate decision.

Check that the run matches the committed code, config and pre-registration (prints `True` three times):

```bash
python -c "import json,sys; sys.path.insert(0,'src'); from pathlib import Path; from statarb.protocol import code_fingerprint, file_sha256; m=json.load(open('docs/research/v3/results/20261024T183817Z_development_v3/run_manifest.json')); print(code_fingerprint([Path('config/research_v3.yaml').resolve()])==m['code_fingerprint']); print(file_sha256('config/research_v3.yaml')==m['config_sha256']); print(file_sha256('docs/research/v3/PREREGISTRATION.md')==m['preregistration_sha256'])"
```
