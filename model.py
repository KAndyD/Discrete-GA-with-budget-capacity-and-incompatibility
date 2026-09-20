"""Бинарный выбор оборудования: бюджет, мощность, несовместимость."""
import numpy as np


def generate(seed):
    rng=np.random.default_rng(seed); n=36
    edges=sorted({tuple(sorted(map(int,rng.choice(n,2,replace=False)))) for _ in range(45)})
    return dict(seed=seed,items=[dict(id=i,name=f'Модуль {i+1:02d}',cost=int(rng.integers(12,50)),power=int(rng.integers(5,35)),utility=int(rng.integers(20,100))) for i in range(n)],conflicts=[list(e) for e in edges])


def arrays(data):
    return tuple(np.array([item[k] for item in data['items']]) for k in ('cost','power','utility'))


def evaluate(x,data,cfg):
    x=np.atleast_2d(x).astype(bool); cost,power,u=arrays(data)
    c=x@cost; p=x@power
    conflict=sum((x[:,a]&x[:,b]).astype(int) for a,b in data['conflicts'])
    violation=np.maximum(c-cfg['budget'],0)+np.maximum(p-cfg['power_limit'],0)+conflict
    value=x@u
    return value,violation


def repair(x,data,cfg):
    """Удаляем менее выгодные объекты до выполнения всех ограничений."""
    x=np.array(x,dtype=bool,copy=True); cost,power,u=arrays(data)
    ratio=u/(cost/cfg['budget']+power/cfg['power_limit'])
    for a,b in data['conflicts']:
        if x[a] and x[b]: x[a if ratio[a]<ratio[b] else b]=False
    for i in np.argsort(ratio):
        if x@cost<=cfg['budget'] and x@power<=cfg['power_limit']: break
        x[i]=False
    return x


def mutate(x,rng,kind,probability):
    x=x.copy()
    if kind=='flip':
        x ^= rng.random(x.shape)<probability
    elif kind=='swap':
        # Одна попытка обмена на потомка; сохраняет число выбранных объектов.
        for row in x:
            a,b=rng.choice(x.shape[1],2,replace=False); row[a],row[b]=row[b],row[a]
    else: raise ValueError(kind)
    return x


def run(cfg,data,seed,mode='repair',mutation='flip'):
    rng=np.random.default_rng(seed); n=cfg['population']; d=len(data['items'])
    pop=rng.random((n,d))<.22
    # Нулевая особь обеспечивает известный допустимый архив во всех режимах.
    pop[0]=False
    penalty=sum(item['utility'] for item in data['items'])+1
    best_value=-1; best=None; hist=[]; feasible=[]; calls=0; scores=None
    for generation in range(cfg['generations']+1):
        if generation:
            if mode=='random': children=rng.random((n,d))<rng.uniform(.05,.6,(n,1))
            else:
                ids=rng.integers(n,size=(2,n,3)); win=ids[np.arange(2)[:,None],np.arange(n)[None,:],scores[ids].argmin(2)]
                a,b=pop[win[0]],pop[win[1]]
                mask=rng.random((n,d))<.5
                children=np.where(mask,a,b)
                children=np.where(rng.random((n,1))<cfg['crossover_probability'],children,a)
                children=mutate(children,rng,mutation,cfg['mutation_probability'])
        else: children=pop
        if mode in ('repair','random'): children=np.array([repair(x,data,cfg) for x in children])
        values,violations=evaluate(children,data,cfg); calls+=n
        child_scores=-values+penalty*violations
        feasible.append(float((violations==0).mean()))
        for i in np.flatnonzero(violations==0):
            if values[i]>best_value: best_value=int(values[i]); best=children[i].copy()
        if generation and mode!='random':
            merged=np.r_[pop,children]; all_scores=np.r_[scores,child_scores]; order=np.argsort(all_scores,kind='stable')[:n]
            pop,scores=merged[order],all_scores[order]
        else: pop,scores=children,child_scores
        hist.append(best_value)
    return best_value,best,hist,calls,float(np.mean(feasible))
