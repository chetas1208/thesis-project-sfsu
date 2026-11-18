import json, sys, math
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, "/home/923873155/Sem 4 Research/SkillLLM_v2_stage2/src")
from skillllm_v2.evals.research import agreement as ag
nan = np.nan
# Krippendorff (2011) "Computing Krippendorff's Alpha-Reliability", 4 observers x 12 units, missing data
K2011 = [[1,2,3,3,2,1,4,1,2,nan,nan,nan],
         [1,2,3,3,2,2,4,1,2,5,nan,3],
         [nan,3,3,3,2,3,4,2,2,5,1,nan],
         [1,2,3,3,2,4,4,1,2,5,1,nan]]
# krippendorff package / Wikipedia example: 3 coders x 15 units
WIKI = [[nan,nan,nan,nan,nan,3,4,1,2,1,1,3,3,nan,3],
        [1,nan,2,1,3,3,4,3,nan,nan,nan,nan,nan,nan,nan],
        [nan,nan,2,1,3,4,4,nan,2,1,1,3,3,nan,4]]
out = {"krippendorff_version": getattr(krippendorff, "__version__", "0.8.1"), "cases": []}
rng = np.random.default_rng(7)
datasets = {"K2011": K2011, "WIKI": WIKI}
for i in range(6):
    m = rng.integers(1, 5, size=(rng.integers(2, 5), rng.integers(8, 40))).astype(float)
    m[rng.random(m.shape) < 0.15] = nan
    datasets[f"rand{i}"] = m.tolist()
for name, d in datasets.items():
    for lvl in ["nominal", "ordinal", "interval", "ratio"]:
        ref = krippendorff.alpha(reliability_data=np.array(d, dtype=float), level_of_measurement=lvl)
        ours = ag.krippendorff_alpha([[None if (isinstance(v, float) and math.isnan(v)) else v for v in r] for r in d], lvl)
        out["cases"].append({"data": name, "level": lvl, "package": float(ref), "ours": float(ours), "absdiff": abs(float(ref) - float(ours))})
# Cohen kappa vs sklearn
for i in range(6):
    a = rng.integers(0, 4, 60).tolist(); b = [x if rng.random() < .6 else int(rng.integers(0, 4)) for x in a]
    for w in [None, "linear", "quadratic"]:
        ref = cohen_kappa_score(a, b, weights=w, labels=[0,1,2,3])
        ours = ag.cohen_kappa(a, b, [0,1,2,3], w)
        out["cases"].append({"data": f"kappa{i}", "level": str(w), "package": float(ref), "ours": float(ours), "absdiff": abs(ref-ours)})
print(json.dumps({"max_absdiff": max(c["absdiff"] for c in out["cases"]),
                  "K2011": {c["level"]: round(c["package"], 6) for c in out["cases"] if c["data"] == "K2011"},
                  "WIKI": {c["level"]: round(c["package"], 6) for c in out["cases"] if c["data"] == "WIKI"}}, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
