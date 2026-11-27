"""Matched short real-game recovery trials using production collectors and updates.

Forks trusted local learner snapshots into distinct diagnostic runs. Preserves
actor/critic/optimizer state but starts fresh replay, RNG, episodes and counters.
Not from-scratch architecture comparisons or final-test evaluations.
"""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,json,sys,os,gzip,pickle,hashlib,time,gc,shutil
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax,jax.numpy as jnp,numpy as np
import strategy,models,protocol,budget_curriculum as curriculum
import strategy_training as training
from env import resource_taxi as env
from evaluate import evaluate
from resource_study import SETTINGS

OLD=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training'
OUT=ROOT/'artifacts/local_food_recovery'
REPORT=ROOT/'reports/local_food_recovery'
CASES=(('dsac_mlp_control','dsac_mlp',3e-6),('dsac_mlp_entropy','dsac_mlp',3e-4),
       ('ppo_mlp_context','ppo_mlp',3e-6),('ppo_gru_context','ppo_gru',3e-6))

def parent_path(kind):
    if kind=='dsac_mlp':
        gates=[json.loads(s) for s in (OLD/kind/'curriculum_gates.jsonl').read_text().splitlines()]
        tick=next(g['physical_transitions'] for g in gates if g['stage']==2 and g['passed'])
        return OLD/kind/f'learner_{tick:09}.pkl.gz'
    return sorted((OLD/kind).glob('learner_*.pkl.gz'))[-1]

def assess(actor,kind,base,folder,label,food_only=False):
    rows=[]
    for name,config,n in (
        ('food_prerequisite',curriculum.evaluation_configuration(curriculum.configuration(base,3)),16),
        ('food_approach',curriculum.evaluation_configuration(curriculum.configuration(base,6)),16),
        ('food_from_home',curriculum.evaluation_configuration(curriculum.configuration(base,9)),16),
        ('standard',replace(base,training_rehearsal=0.,exploration_epsilon=0.),16)):
        if food_only and name not in ('food_prerequisite','food_approach'):continue
        params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
        summary,states,returns=evaluate(actor,kind,params,range(1000001,1000001+n),interface='strategy')
        counts=np.asarray(states.counters);success=np.asarray(states.outcome)==env.SUCCESS
        row={'context':name,'summary':summary,
             'successful_meal_episodes':int((success&(counts[:,env.MEALS]>0)).sum())}
        rows.append(row)
        np.savez_compressed(folder/(label+'_'+name+'.npz'),outcome=np.asarray(states.outcome),
            cash=np.asarray(states.cash),satiety=np.asarray(states.satiety),counters=counts,
            time=np.asarray(states.time),external_return=np.asarray(returns))
        print('ASSESS',kind,label,name,summary['ownership'],'/',n,'meals',summary['meals'],flush=True)
    (folder/(label+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows

def run(name,kind,alpha,decisions,head_scale=1.,food_only=False,parent_override=None,wait_ticks=1,actor_update='reverse_kl'):
    if head_scale!=1.:name+=f'_head{head_scale:g}'
    if food_only:name+='_foodonly'
    if parent_override:name+='_continued'
    if wait_ticks!=1:name+=f'_wait{wait_ticks}'
    if actor_update!='reverse_kl':name+='_'+actor_update
    folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    if (folder/'plan.json').exists():raise RuntimeError('Interrupted trial exists; preserve it and use a new registered name')
    parent=(ROOT / parent_override).resolve() if parent_override else parent_path(kind)
    with gzip.open(parent,'rb') as f:saved=pickle.load(f)
    meta=saved.get('metadata')
    if meta is None:
        _,meta=models.load_policy(parent.parent/'final.npz')
        saved['total']=json.loads((parent.parent/'result.json').read_text())['physical_ticks']
    base=env.Config(**meta['evaluation_environment'])
    base=replace(base,loan_capacity=519,unavailable_wait_ticks=wait_ticks)
    plan={'name':name,'kind':kind,'parent':parent.name,
        'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'parent_physical_ticks':saved['total'],'source_hashes':protocol.source_hashes(),
        'network_decisions':decisions,'alpha_d':alpha,'wait_ticks':wait_ticks,'seed':301,
        'categorical_head_scale':head_scale,'actor_optimizer_reset':head_scale!=1.,
        'categorical_actor_update':actor_update,
        'training_scenarios':[11,10010],'development_scenarios':[1000001,1000016],
        'phases':([3] if food_only else [3,7,8,9,16] if kind.startswith('dsac') else [7,8,9,16]),
        'scope':'Warm-fork diagnostic; no replay inherited, no final test, no demonstrations or reward changes.'}
    (folder/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    shutil.copy2(__file__,folder/'trial_source.py')
    actor=jax.tree.map(jnp.asarray,saved['actor'])
    is_dsac=kind.startswith('dsac')
    learner=jax.tree.map(jnp.asarray,saved['learner']) if is_dsac else None
    opt_state=None if is_dsac else jax.tree.map(jnp.asarray,saved['opt_state'])
    del saved;gc.collect()
    if is_dsac:
        cfg=training.dsac.AgentConfig(gamma=base.gamma,demo_weight=0.,action_count=strategy.COUNT,
            **{**SETTINGS,'discrete_temperature':alpha,'categorical_actor_update':actor_update})
        agent=training.dsac_agent(kind,cfg)
        learner=learner._replace(log_temperatures=jnp.log(jnp.array((cfg.continuous_temperature,alpha))))
        update=training.make_dsac_update(agent,warmup=2048)
    if head_scale!=1.:
        # Flatten all economic action logits equally, with no preferred action.
        # Continuous controls and PPO's value output retain their weights.
        if is_dsac:
            w,b=actor[-1]
            actor=(*actor[:-1],(w.at[:,4:].multiply(head_scale),b.at[4:].multiply(head_scale)))
            learner=learner._replace(actor=actor,actor_opt=agent.actor_optimizer.init(actor))
        else:
            w,b=actor['head'][-1]
            head=(w.at[:,4:-1].multiply(head_scale),b.at[4:-1].multiply(head_scale))
            actor={**actor,'head':(*actor['head'][:-1],head)}
            opt_state=training.OPT.init(actor)
    before=assess(actor,kind,base,folder,'before',food_only)
    replay=None;cursor=jnp.int32(0);size=jnp.int32(0)
    rng=jax.random.PRNGKey(301);physical=0;started=time.monotonic();previous=None
    phases=plan['phases'];chunks=decisions//512
    for index in range(chunks):
        phase=phases[min(index*len(phases)//chunks,len(phases)-1)]
        if phase!=previous:
            config=curriculum.configuration(base,phase)
            params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
            counters=jnp.arange(training.ENVS)
            obs,states=training.reset_batch(params,protocol.load(),counters)
            hidden=models.initial_memory(kind,training.ENVS,obs.features.shape[-1])
            collector=training.compiled_collector(config,kind,protocol.load()['training_scenario_count'])
            if not is_dsac:update=training.compiled_ppo(config)
            previous=phase
        rng,collect_key,update_key=jax.random.split(rng,3)
        (states,hidden,counters,_),trace=collector(actor,states,hidden,counters,collect_key,None)
        if np.asarray(trace[5]['capacity_needed']).any():raise RuntimeError('Insufficient loan storage')
        if is_dsac:
            transition=trace[0]
            if replay is None:
                replay=jax.tree.map(lambda x:jnp.zeros((training.CAPACITY,*x.shape[2:]),x.dtype),transition)
            learner,replay,cursor,size,loss=update(learner,replay,cursor,size,transition,update_key)
            actor=learner.actor
        else:actor,opt_state,loss=update(actor,opt_state,trace,states,hidden,update_key)
        losses=np.asarray(loss)
        if not np.isfinite(losses).all():raise FloatingPointError('Nonfinite recovery loss')
        p=trace[5];active=np.asarray(p['active']);physical+=int(active.sum())
        counts=np.asarray(p['events'])[active].sum(axis=0)
        row={'decisions':(index+1)*512,'physical_ticks':physical,'stage':phase,
             'loss':losses.tolist(),'completed_actions':dict(zip(env.COUNTER_NAMES,map(int,counts))),
             'elapsed_seconds':time.monotonic()-started}
        with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('TRAIN',name,row['decisions'],'/',decisions,'stage',phase,'meals',row['completed_actions']['meals'],flush=True)
    models.save_policy(folder/'final.npz',actor,{**meta,'source_hashes':protocol.source_hashes(),
        'local_recovery_plan':plan,'diagnostic_only':True,'environment':asdict(base),'evaluation_environment':asdict(base)})
    training.checkpoint(folder/'learner.pkl.gz',{'actor':actor,'learner':learner,'opt_state':opt_state,
        'metadata':{**meta,'local_recovery_plan':plan},'total':physical,'plan':plan})
    after=assess(actor,kind,base,folder,'after',food_only)
    passed=not food_only and all(x['summary']['ownership']>=12 for x in after) and all(
        x['successful_meal_episodes']>=12 for x in after if x['context']!='standard')
    result={'name':name,'kind':kind,'before':before,'after':after,'passed_short_development':passed,
        'passed_food_acquisition':all(x['summary']['ownership']>=12 and x['successful_meal_episodes']>=12
            for x in after if x['context']=='food_prerequisite'),
        'passed_food_transfer':all(x['summary']['ownership']>=12 and x['successful_meal_episodes']>=12
            for x in after if x['context']=='food_approach'),
        'network_decisions':decisions,'physical_ticks':physical,
        'scope':'16-episode development checks. Passing requires confirmation on 64 episodes and both retained banks before cluster promotion.'}
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TRIAL_RESULT',json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=[x[0] for x in CASES]+['all'],default='all')
    parser.add_argument('--decisions',type=int,default=16384)
    parser.add_argument('--head-scale',type=float,choices=(.2,1.),default=1.)
    parser.add_argument('--food-only',action='store_true')
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--wait-ticks',type=int,choices=(1,8),default=1)
    parser.add_argument('--actor-update',choices=('reverse_kl','forward_kl'),default='reverse_kl')
    args=parser.parse_args()
    if args.parent and args.case=='all':raise ValueError('A parent override requires a single architecture')
    if args.decisions<2560 or args.decisions%512:raise ValueError('Use at least 2560 decisions in complete 512-decision chunks')
    REPORT.mkdir(parents=True,exist_ok=True)
    print('DEVICES',jax.devices(),flush=True)
    results=[]
    for name,kind,alpha in CASES:
        if args.case not in ('all',name):continue
        results.append(run(name,kind,alpha,args.decisions,args.head_scale,args.food_only,args.parent,args.wait_ticks,args.actor_update))
        suffix='' if args.head_scale==1. and not args.food_only and not args.parent and args.wait_ticks==1 else '_followup'
        if args.actor_update!='reverse_kl':suffix+='_'+args.actor_update
        (REPORT/('results'+suffix+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
        jax.clear_caches();gc.collect()
    print('LOCAL_TRIALS_DONE',flush=True)

if __name__=='__main__':main()
