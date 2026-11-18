Annotators return their filled packet here as `<packet>__A.csv` / `<packet>__B.csv` (unchanged item_id column).
The adjudicator returns `<packet>__ADJ.csv` containing only the items listed in `<packet>__ADJ_todo.csv`.
Then run `PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples --score`.
