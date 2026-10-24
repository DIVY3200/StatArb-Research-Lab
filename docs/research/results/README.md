# Experiment results (committed copies)

The experiment runner writes to `reports/experiments/`, which is not version-controlled. These folders are
**byte-identical copies** of the three v2 runs, committed so the evidence behind
[`../RESEARCH_REPORT.md`](../RESEARCH_REPORT.md) is available from a clone. Paths such as
`reports/experiments/20261021T162712Z_development` in `freeze.json`, `holdout_evaluated.json` and the report refer
to the folder of the same name here.

| Folder | Stage | Notes |
|---|---|---|
| `20261021T161656Z_development` | Development, 2015–2024 | First run. Superseded after amendments A1 and A2 (`../PREREGISTRATION.md` §8); kept for transparency. |
| `20261021T162712Z_development` | Development, 2015–2024 | The **frozen** development run (`../freeze.json`) |
| `20261021T162902Z_final` | One-time holdout, 2025-01-01 to 2026-09-30 | `decision.json`: **reject the tested hypothesis** |

Each folder contains:

* `run_manifest.json`: stage, timestamps, code, config and pre-registration fingerprints, input data hashes, special sessions dropped, environment, seeds
* `config_snapshot.yaml`: the exact configuration used
* `summary.json`: pooled metrics, bootstrap intervals, classification diagnostics, costs and regimes
* `variant_metrics_*.csv`, `windows_*.csv`, `trades_*.csv`, `predictions_*.csv`, `net_returns_*.csv`, `pooled_net_returns.csv`
* `decision.json`: final run only

**Verify the freeze chain.** The SHA-256 of `20261021T162712Z_development/summary.json` equals
`development_summary_sha256` in `../freeze.json`:

```bash
python -c "import hashlib,json; f=json.load(open('docs/research/freeze.json')); print(hashlib.sha256(open('docs/research/results/20261021T162712Z_development/summary.json','rb').read()).hexdigest()==f['development_summary_sha256'])"
```

Raw prices are not redistributed. Re-running the experiments downloads them again from Yahoo Finance, and the
providers' data may have been revised since, so check the input hashes in each `run_manifest.json`.
