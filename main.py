"""Сравнение ограничений и операторов на фиксированном наборе оборудования."""
import json
import time
import numpy as np
from common import ROOT,setup,write_json,write_csv,summarize,convergence,save_histories,style,plt
from model import generate,run,evaluate,arrays


def main():
    cfg,out=setup(); path=ROOT/'data/equipment.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else generate(cfg['data_seed'])
    if not path.exists(): write_json(path,data)
    write_json(out/'input_data.json',data)
    rows=[]; histories={}; best={}
    for method,mode,mutation in [('repair_flip','repair','flip'),('penalty_flip','penalty','flip'),('repair_swap','repair','swap'),('random_repair','random','flip')]:
        histories[method]=[]
        for seed in range(cfg['seed'],cfg['seed']+cfg['runs']):
            start=time.perf_counter(); value,x,h,calls,feasible=run(cfg,data,seed,mode,mutation)
            rows.append(dict(method=method,seed=seed,best=value,feasible_fraction=feasible,evaluations=calls,seconds=time.perf_counter()-start)); histories[method].append(h)
            if method not in best or value>best[method]['utility']:
                cost,power,_=arrays(data)
                best[method]=dict(utility=value,seed=seed,selected=np.flatnonzero(x).tolist(),chromosome=x.astype(int).tolist(),cost=int(x@cost),power=int(x@power))
        print(method,'finished',flush=True)
    write_csv(out/'runs.csv',rows); write_csv(out/'summary.csv',summarize(rows,['best','feasible_fraction','seconds','evaluations'])); write_json(out/'best.json',best)
    save_histories(out/'histories.csv',histories); convergence(out/'convergence.png',histories,ylabel='Полезность допустимого решения')
    # Два допустимых и два недопустимых примера, с фактическими нарушениями.
    d=len(data['items']); examples=[]
    for name,ids in [('empty',[]),('single',[0]),('all',list(range(d))),('conflict_pair',data['conflicts'][0])]:
        x=np.zeros(d,dtype=bool); x[ids]=True; value,violation=evaluate(x,data,cfg)
        examples.append(dict(name=name,selected=ids,utility=int(value[0]),violation=int(violation[0]),feasible=bool(violation[0]==0)))
    write_json(out/'constraint_examples.json',examples)
    chosen=max(best.values(),key=lambda b:b['utility']); write_csv(out/'selected_equipment.csv',[data['items'][i] for i in chosen['selected']])
    style(); fig,ax=plt.subplots(figsize=(10,5)); items=[data['items'][i] for i in chosen['selected']]
    ax.bar([str(x['id']+1) for x in items],[x['utility'] for x in items],color='#087f8c'); ax.set(xlabel='Номер модуля',ylabel='Полезность',title=f"Лучший комплект: стоимость {chosen['cost']}, мощность {chosen['power']}")
    fig.tight_layout(); fig.savefig(out/'equipment.png'); plt.close(fig)

if __name__=='__main__': main()
