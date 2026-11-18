# L11 — Human evaluation (packets built; human metrics pending)

Protocol: `docs/thesis/evaluation/human_eval_protocol.md`. Two independent annotators (A, B) + one adjudicator; system identity hidden; item order randomised separately per annotator; answer keys live in `inputs/evals/research/L11_human_eval/answer_keys/` (never handed out). LLM judges are not used.

Records: {'measured': 8, 'pending_human': 49}. Every human metric is `pending_human` until both annotators return >= 90% of a packet.

## Packets

| packet | title | items | origin |
|---|---|---|---|
| P1 | Extraction: candidate span validity, type and boundary | 201 | SkillSpan test |
| P2 | Extraction: full-sentence span annotation | 100 | SkillSpan test |
| P3 | Normalization / taxonomy linking | 150 | reports/skillllm_v2/evals/L06_linking/annotation_sample.csv |
| P4a | Graph: prerequisite edge validity | 80 | inputs/courses/course_catalog_large.jsonl (synthetic, project-authored catalogue) |
| P4b | Graph: learning-path pedagogical validity | 60 | reports/skillllm_v2/evals/L08_graph_retrieval/paths_for_human_eval.jsonl |
| P5 | Course recommendation: relevance, gap closure, readiness | 213 | cached re-run of the frozen DEFAULT_CONFIG recommender inputs/evals/research/L11_human_eva |
| P6 | Course recommendation: explanation faithfulness | 120 | cached re-run of the frozen DEFAULT_CONFIG recommender inputs/evals/research/L11_human_eva |
| P7 | Robustness: perturbation validity | 50 | reports/skillllm_v2/evals/L03_robustness/perturbation_validity_sample.csv |

Training round T0 (synthetic items with answer key and rationale): {'P1': 8, 'P2': 4, 'P3': 4, 'P4a': 3, 'P5': 2}.

## Sample-size justification

| target | halfwidth | n_required | basis |
|---|---|---|---|
| proportion within one P1 stratum (n=50) | 0.139 | 50 | Wald, p=0.5 worst case |
| pooled proportion | 0.100 | 97 | Wald, p=0.5 worst case |
| pooled proportion | 0.070 | 196 | Wald, p=0.5 worst case |
| pooled proportion | 0.050 | 385 | Wald, p=0.5 worst case |
| Cohen kappa (expected 0.6, prevalence 0.5) | 0.100 | 246 | Cohen (1960) large-sample SE |
| Cohen kappa simulated at n=100 (kappa 0.6, prev 0.5) | 0.158 | 100 | Monte-Carlo, 2000 draws, seed 42 |
| Cohen kappa simulated at n=150 (kappa 0.6, prev 0.5) | 0.128 | 150 | Monte-Carlo, 2000 draws, seed 42 |
| Cohen kappa simulated at n=200 (kappa 0.6, prev 0.5) | 0.116 | 200 | Monte-Carlo, 2000 draws, seed 42 |
| mean relevance 0-3 (sd 1.0) | 0.150 | 171 | normal approx. |

## Design notes

- P1 strata come from the SkillSpan test gold and stored predictions (974-document anchor subset). The anchor_fp stratum measures gold incompleteness: an anchor 'false positive' that humans accept as a skill means the reported precision is pessimistic. The control stratum (agreed true positives) checks annotator calibration.
- P2 includes gold-free test sentences, the population excluded from all stored extraction predictions (974/3,569 caveat), so the human rate of skills in 'skill-free' sentences can be set against A3's spurious-prediction rate.
- P5 annotators see the job text and the listed required/preferred skills (the systems' inputs), NOT the rule-label `gold.needs` (which include implicit needs and drive the v1 rule labels); the legacy 60-row sample showed its automatic grade and needed-skill weights and is re-blinded here.
- P6 cannot be system-blind (only the Full recommender produces explanations) and is reported as such.
- Agreement: Cohen's kappa (quadratic-weighted for ordinal scales), Gwet's AC1 for nominal scales, Krippendorff's alpha at the scale's level, span-level F1 for free span annotation; 95% percentile bootstrap over items. `agreement.py` reproduces the published Krippendorff (2011) example (nominal 0.743, ordinal 0.815, interval 0.849, ratio 0.797) and matches the `krippendorff` package and scikit-learn to <= 2.3e-16 (see `agreement_oracle_check.json`).

Command: `PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples --seed 42 --n-boot 2000` (build); add `--score` after annotations are returned.
