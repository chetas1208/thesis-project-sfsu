# Reports Directory

Generated artifacts are written here:

- `reports/metrics/*.json`: machine-readable metric summaries
- `reports/tables/*.csv`: paper-friendly tables
- `reports/tables/*.tex`: LaTeX table exports
- `reports/plots/*.png`: plots and latency charts
- `reports/errors/*.csv`: categorized error analysis

The summary report files at the root of this directory are refreshed by:

```bash
python -m src.eval.make_latex_tables
```
