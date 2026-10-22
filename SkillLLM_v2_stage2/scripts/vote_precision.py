import sys, json
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from collections import defaultdict
from skillllm_v2.eval import significance as S
import ensemble_eval as E
docs=S.load_gold_docs(); ids=[d['id'] for d in docs]
gold=[d['gold_spans'] if 'gold_spans' in d else d.get('spans',[]) for d in docs]
preds={n:S.load_system_predictions(p,ids) for n,p in E.SYSTEMS.items()}
sysl=[preds['anchor'],preds['v1'],preds['v2']]
tot=defaultdict(lambda:[0,0]); per_sys=defaultdict(lambda:[0,0])
for i in range(len(ids)):
    gk={S._key(g) for g in gold[i]}
    votes=defaultdict(int)
    for s in sysl:
        seen=set()
        for sp in s[i]:
            k=S._key(sp)
            if k in seen: continue
            seen.add(k); votes[k]+=1
    for k,v in votes.items():
        tot[v][0]+= (k in gk); tot[v][1]+=1
res={v:(c,n,c/n) for v,(c,n) in sorted(tot.items())}
tp_all=sum(c for c,n in tot.values()); n_all=sum(n for c,n in tot.values())
gold_total=sum(len({S._key(g) for g in gg}) for gg in gold)
print("votes -> correct/total, precision")
for v,(c,n,p) in res.items(): print(v,c,n,round(p,4))
cum=0;cn=0
print("\ncumulative (keep spans with >=v votes): precision, recall")
for v in (3,2,1):
    c=sum(tot[x][0] for x in tot if x>=v); n=sum(tot[x][1] for x in tot if x>=v)
    print(v, round(c/n,4), round(c/gold_total,4), "F1", round(2*(c/n)*(c/gold_total)/((c/n)+(c/gold_total)),4))
