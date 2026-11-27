"""Short sequential checks of observations, replay and the tuned learners."""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import os
import pickle
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import budget_curriculum as curriculum
from env import resource_taxi as env
from evaluate import evaluate
import models
import protocol
import strategy
import strategy_training as train
from resource_study import SETTINGS

OUT=ROOT/'reports/local_food_recovery/smoke'
OLD=(Path(os.environ['GAMBLERS_TAXI_RECORDED_RUNS']) if 'GAMBLERS_TAXI_RECORDED_RUNS' in os.environ else ROOT / 'artifacts/recorded_runs') / 'resource_study/output/artifacts/resource_study/training'
REFERENCE=OLD/'dsac_gru/selected.npz'
IDS=range(1000001,1000017)
DESIGN=protocol.load()


def save(name,result):
    OUT.mkdir(parents=True,exist_ok=True)
    record={'test':name,**result}
    path=OUT/(name+'.json')
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2),encoding='utf-8')
    tmp.replace(path)
    print(json.dumps(record),flush=True)


def params(config):
    return env.ResourceTaxiEnv.init_params(jax.random.key(0),config)[0]


def base_config():
    _,meta=models.load_policy(REFERENCE)
    return env.Config(**meta['evaluation_environment'])


def obs_check():
    base=replace(base_config(),observation_history=True,resource_observation=True)
    p=params(curriculum.evaluation_configuration(curriculum.configuration(base,3)))
    reset=eqx.filter_jit(strategy.reset)
    advance=eqx.filter_jit(strategy.advance)
    get_raw=eqx.filter_jit(lambda s:strategy.decision_features(env.ResourceTaxiEnv.get_observation(s,p).features,s,p))
    for seed in range(11,19):
        obs,state=reset(seed,p)
        assert obs.features.shape==(370,)
        np.testing.assert_array_equal(obs.features[-26:],np.zeros(26))
        raw=get_raw(state)
        key=jax.random.fold_in(jax.random.key(301),seed)
        for _ in range(160):
            key,action_key=jax.random.split(key)
            choice=jax.random.categorical(action_key,jnp.where(obs.action_mask,0.,-jnp.inf)).astype(jnp.int32)
            next_obs,state,_,done,_=advance(state,env.Action(jnp.zeros(2),choice),p)
            current=get_raw(state)
            np.testing.assert_array_equal(next_obs.features,jnp.concatenate((current,raw,jax.nn.one_hot(choice,26))))
            obs,raw=next_obs,current
            if bool(done):break
        if bool(done):break
    assert bool(done)
    # Labels are only used by this supervised wiring check and never exported.
    obs,state=reset(11,p)
    state=replace(state,satiety=jnp.float32(.35))
    obs=strategy.get_observation(state,p)
    features,masks,labels=[],[],[]
    completed=np.zeros(14,dtype=int)
    for choice in [strategy.RESTAURANT,strategy.FUEL_SMALL]+[strategy.WAIT]*30:
        assert bool(obs.action_mask[choice])
        features.append(obs.features);masks.append(obs.action_mask);labels.append(choice)
        obs,state,_,done,trace=advance(state,env.Action(jnp.zeros(2),jnp.int32(choice)),p)
        completed+=np.asarray(trace['events'])[np.asarray(trace['active'])].sum(0)
        assert not bool(done)
    assert completed[env.MEALS]>0 and completed[env.REFILLS]>0
    x,m,y=jnp.stack(features),jnp.stack(masks),jnp.array(labels)
    actor=models.init_ppo(jax.random.key(301),x.shape[-1],False,strategy.COUNT)
    optimizer=optax.adam(.003)
    def logits(a):return models.ppo_step(a,x,m,jnp.zeros((32,models.WIDTH)))[2]
    @jax.jit
    def update(a,s):
        grad=jax.grad(lambda b:-logits(b)[jnp.arange(32),y].mean())(a)
        delta,s=optimizer.update(grad,s)
        return optax.apply_updates(a,delta),s
    state_opt=optimizer.init(actor)
    for _ in range(200):actor,state_opt=update(actor,state_opt)
    correct=int((logits(actor).argmax(-1)==y).sum())
    save('test1',{'passed':correct==32,'accuracy':correct/32,'examples':32,
        'correct':correct,'completed_meals':int(completed[env.MEALS]),'completed_refills':int(completed[env.REFILLS]),
        'random_episode_completed':True,'exact_history_and_zero_reset_action':True})


def replay_check():
    config=replace(base_config(),standard_start_rehearsal=.25)
    p=params(curriculum.configuration(config,9))
    actor,meta=models.load_policy(REFERENCE)
    counters=jnp.arange(train.ENVS)
    obs,states=train.reset_batch(p,DESIGN,counters)
    hidden=models.initial_memory('dsac_gru',train.ENVS,obs.features.shape[-1])
    (_,_,_,_),trace=train.make_collector(p,'dsac_gru',DESIGN)(
        actor,states,hidden,counters,jax.random.key(301),None)
    # Keep one representative reset per scenario ID from the actual collector.
    physical=trace[5]
    ids=np.asarray(physical['scenario_id'])
    times=np.asarray(physical['time'])
    active=np.asarray(physical['active'])
    debt=np.asarray(physical['car_debt'])
    starts={}
    for sid,amount in zip(ids[active&(times==1)],debt[active&(times==1)]):
        starts[int(sid)]=float(amount)>300.
    _,all_starts=eqx.filter_jit(jax.vmap(strategy.reset,in_axes=(0,None)))(jnp.arange(11,111),p)
    standard=np.asarray(all_starts.car_debt)==500.
    assert np.allclose(np.asarray(all_starts.agent_pos)[standard],p.zone_centers[0])
    assert np.all(np.asarray(all_starts.car_debt)[~standard]==150.)
    p=params(env.Config(**meta['evaluation_environment']))
    a,sa,ra=evaluate(actor,'dsac_gru',p,range(1000001,1000065),deterministic=True,interface='strategy')
    b,sb,rb=evaluate(actor,'dsac_gru',p,range(1000001,1000065),deterministic=True,interface='strategy')
    same=all(x.dtype==y.dtype and x.shape==y.shape and np.asarray(x).tobytes()==np.asarray(y).tobytes()
             for x,y in zip(jax.tree.leaves((sa,ra)),jax.tree.leaves((sb,rb)))) and a==b
    share=float(standard.mean())
    save('test4',{'passed':same and .15<=share<=.35,'standard_starts':int(standard.sum()),
         'reset_sample_size':100,'collector_starts_seen':len(starts),
         'collector_network_decisions':512,'bitwise_equal':same,'wins':[a['ownership'],b['ownership']]})


def measure(actor,kind,config,ids=IDS):
    summary,states,_=evaluate(actor,kind,params(config),ids,deterministic=True,interface='strategy')
    return {'wins':summary['ownership'],'starvation':summary['starvation'],
        'meals_per_episode':summary['meals'],'refills_per_episode':summary['refills'],
        'terminal_at_zero_fuel':int(((np.asarray(states.fuel)<=0)&
              np.isin(np.asarray(states.outcome),(env.DEATH,env.CRASH))).sum())}


def dsac_check(stacked=False):
    parent=ROOT/'artifacts/local_food_recovery/dsac_mlp_control/learner.pkl.gz'
    with gzip.open(parent,'rb') as f:learner=pickle.load(f)['learner']
    learner=jax.tree.map(jnp.asarray,learner)
    learner=learner._replace(log_temperatures=learner.log_temperatures.at[1].set(jnp.log(.05)))
    base=base_config()
    if stacked:
        base=replace(base,resource_observation=True,observation_history=True)
        # Map the original current observation and previous action to their
        # exact new coordinates. Start the new input weights at zero.
        index=jnp.concatenate((jnp.arange(168),jnp.arange(344,370)))
        def lift(network):
            w,b=network[0]
            trailing=w.shape[0]-194
            destination=jnp.concatenate((index,jnp.arange(370,370+trailing)))
            expanded=jnp.zeros((370+trailing,w.shape[1]),w.dtype).at[destination].set(w)
            return ((expanded,b),*network[1:])
        actor=lift(learner.actor)
        critics=tuple(map(lift,learner.critics))
        targets=tuple(map(lift,learner.targets))
        optimizer=train.dsac_agent('dsac_mlp',train.dsac.AgentConfig())
        learner=learner._replace(actor=actor,critics=critics,targets=targets,
            actor_opt=optimizer.actor_optimizer.init(actor),critic_opt=optimizer.opt.init(critics),
            temperature_opt=optimizer.temp_opt.init(learner.log_temperatures))
    contexts=(1,3,10)
    probe=[]
    for stage in (10,1):
        p=params(curriculum.evaluation_configuration(curriculum.configuration(base,stage)))
        obs,_=eqx.filter_jit(jax.vmap(strategy.reset,in_axes=(0,None)))(jnp.arange(1000001,1000017),p)
        probe.append((obs.features,obs.action_mask))
    def confidence(a):
        rows=[]
        for name,(x,mask) in zip(('low_fuel','bank'),probe):
            _,_,logp=train.dsac.actor_outputs(a,x,mask)
            probability=jnp.exp(logp)
            rows.append({'context':name,'entropy':float(-(probability*logp).sum(-1).mean()),
                'minimum_max_probability':float(probability.max(-1).min()),
                'median_max_probability':float(jnp.median(probability.max(-1)))})
        return rows
    initial=confidence(learner.actor)
    # Populate replay with real simulator transitions and keep replay at the switch.
    # The legacy checkpoint requires its original 194-feature input.
    collectors={}
    replay=None;cursor=jnp.int32(0);size=jnp.int32(0)
    rng=jax.random.key(301)
    replay_decisions=0
    if stacked:
        # A fixed batch of actual transitions isolates the two actor updates
        # and avoids collecting another 52k decisions for a gradient smoke.
        transitions=[]
        for stage in (*contexts,16):
            p=params(curriculum.configuration(base,stage))
            c=jnp.arange(train.ENVS)
            obs,s=train.reset_batch(p,DESIGN,c)
            h=models.initial_memory('dsac_mlp',train.ENVS,obs.features.shape[-1])
            collect=train.make_collector(p,'dsac_mlp',DESIGN)
            for _ in range(5):
                rng,key=jax.random.split(rng)
                (s,h,c,_),trace=collect(learner.actor,s,h,c,key,None)
                transitions.append(jax.tree.map(lambda x:x.reshape((-1,*x.shape[2:])),trace[0]))
        data=jax.tree.map(lambda *arrays:jnp.concatenate(arrays,axis=0),*transitions)
        replay_decisions=len(data.reward)
        print('STACKED_REPLAY_DECISIONS',replay_decisions,flush=True)
        for method in ('forward_kl','reverse_kl'):
            cfg=train.dsac.AgentConfig(gamma=base.gamma,action_count=strategy.COUNT,
                **{**SETTINGS,'discrete_temperature':.05,'categorical_actor_update':method,
                   'automatic_discrete_temperature':True,'discrete_target_entropy_fraction':.2})
            agent=train.dsac_agent('dsac_mlp',cfg)
            def batch_update(current,key):
                def step(current,step_key):
                    sample,update=jax.random.split(step_key)
                    idx=jax.random.randint(sample,(cfg.minibatch_size,),0,replay_decisions)
                    return agent.update(current,update,jax.tree.map(lambda x:x[idx],data))
                current,loss=jax.lax.scan(step,current,jax.random.split(key,50))
                return current,loss.mean(0)
            run_block=eqx.filter_jit(batch_update)
            for _ in range(50):
                rng,key=jax.random.split(rng)
                learner,loss=run_block(learner,key)
                assert np.isfinite(np.asarray(loss)).all()
            print(method,'2500 updates','alpha_d',float(jnp.exp(learner.log_temperatures[1])),
                  confidence(learner.actor),flush=True)
    for chunk in range(0 if stacked else 103):
        stage=contexts[(chunk//9)%len(contexts)]
        if stage not in collectors:
            config=curriculum.configuration(base,stage)
            p=params(config)
            counters=jnp.arange(train.ENVS)
            obs,s=train.reset_batch(p,DESIGN,counters)
            hidden=models.initial_memory('dsac_mlp',train.ENVS,obs.features.shape[-1])
            collectors[stage]=(train.make_collector(p,'dsac_mlp',DESIGN),s,hidden,counters)
        collector,s,h,c=collectors[stage]
        rng,a,b=jax.random.split(rng,3)
        (s,h,c,_),trace=collector(learner.actor,s,h,c,a,None)
        collectors[stage]=(collector,s,h,c)
        transition=trace[0]
        if replay is None:
            replay=jax.tree.map(lambda x:jnp.zeros((train.CAPACITY,*x.shape[2:]),x.dtype),transition)
        method='forward_kl' if chunk<53 else 'reverse_kl'
        cfg=train.dsac.AgentConfig(gamma=base.gamma,action_count=strategy.COUNT,
            **{**SETTINGS,'gradient_steps':50,'discrete_temperature':.05,'categorical_actor_update':method,
               'automatic_discrete_temperature':stacked,'discrete_target_entropy_fraction':.2 if stacked else .55})
        update=train.make_dsac_update(train.dsac_agent('dsac_mlp',cfg),warmup=2048)
        learner,replay,cursor,size,loss=update(learner,replay,cursor,size,transition,b)
        assert np.isfinite(np.asarray(loss)).all()
        if chunk in (52,102):
            print('DSAC updates',50*(chunk-2),'alpha_d',float(jnp.exp(learner.log_temperatures[1])),
                  confidence(learner.actor),flush=True)
    after=confidence(learner.actor)
    stage3=curriculum.evaluation_configuration(curriculum.configuration(base,3))
    ownership_stage=curriculum.evaluation_configuration(curriculum.configuration(base,0))
    skills=measure(learner.actor,'dsac_mlp',stage3)
    campaign=measure(learner.actor,'dsac_mlp',base)
    first=measure(learner.actor,'dsac_mlp',ownership_stage)
    passed=(all(x['entropy']<.35 and x['minimum_max_probability']>.85 for x in after)
        and 1<=campaign['meals_per_episode']<=2 and campaign['terminal_at_zero_fuel']==0)
    save('test2_stacked' if stacked else 'test2',{'passed':passed,'updates':5000,
        'initial_alpha_discrete':.05,'final_alpha_discrete':float(jnp.exp(learner.log_temperatures[1])),
        'fixed_replay_decisions':replay_decisions if stacked else None,
        'target_entropy_fraction':.2 if stacked else None,'initial':initial,'final':after,
        'first_curriculum_stage':first,'food_prerequisite':skills,'standard':campaign,
        'scope':'Warm fork adapted to 370 features with real two-frame observations and fresh optimizer states.'
        if stacked else 'Warm fork using historical 194-feature observations.'})


def ppo_check(kind):
    recurrent=kind=='ppo_gru'
    config=replace(base_config(),resource_observation=True,observation_history=not recurrent,
        ppo_native_exploration=True,ppo_exploration_decay=True,ppo_epochs=12,
        ppo_target_kl=.02,ppo_clip=.15,ppo_normalize_value_loss=True,
        ppo_value_head_lr_multiplier=3. if recurrent else 1.)
    p=params(config)
    counters=jnp.arange(train.ENVS)
    obs,states=train.reset_batch(p,DESIGN,counters)
    rng,init=jax.random.split(jax.random.key(301))
    actor=models.init_ppo(init,obs.features.shape[-1],recurrent,strategy.COUNT)
    hidden=models.initial_memory(kind,train.ENVS,obs.features.shape[-1])
    optimizer_state=train.OPT.init(actor)
    collector=train.make_collector(p,kind,DESIGN)
    update=train.make_ppo_update(p,train.OPT,diagnostics=True)
    baseline=measure(actor,kind,config)
    history=[]
    for chunk in range(32):
        rng,a,b=jax.random.split(rng,3)
        (states,hidden,counters,_),trace=collector(actor,states,hidden,counters,a,None)
        actor,optimizer_state,metrics=update(actor,optimizer_state,trace,states,hidden,b)
        loss,count,kl,ev=map(float,np.asarray(metrics))
        assert np.isfinite((loss,count,kl,ev)).all()
        history.append({'decisions':(chunk+1)*512,'loss':loss,'optimizer_updates':int(count),
            'approx_kl':kl,'explained_variance':ev})
        if chunk in (15,18,31):print(kind,history[-1],flush=True)
    after=measure(actor,kind,config)
    ev=max(r['explained_variance'] for r in history if 8000<=r['decisions']<=10000)
    kl=max(r['approx_kl'] for r in history)
    stops=[r for r in history if r['optimizer_updates']<96]
    target_stops=[r for r in stops if 40<=r['optimizer_updates']<80]
    acceptable=ev>.6 and .0001<=kl<=.05 and bool(target_stops) and after['wins']>=4 and after['wins']>baseline['wins']
    save('test3_'+kind,{'passed':acceptable,'before':baseline,'after':after,'max_ev_8k_to_10k':ev,
        'max_approx_kl':kl,'early_stops':len(stops),'early_stops_epochs_6_to_10':len(target_stops),'history':history})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    order=('obs','replay','dsac','ppo_mlp','ppo_gru')
    parser.add_argument('test',choices=('all','dsac_stacked',*order))
    args=parser.parse_args()
    if args.test=='all':
        errors=[]
        for name in order:
            print('START',name,flush=True)
            result=subprocess.run([sys.executable,'-B',__file__,name],check=False)
            if result.returncode:errors.append(name)
        if errors:raise SystemExit('Execution errors: '+', '.join(errors))
        return
    checks={'obs':obs_check,'replay':replay_check,'dsac':dsac_check,'dsac_stacked':lambda:dsac_check(stacked=True)}
    if args.test in checks:checks[args.test]()
    else:ppo_check(args.test)


if __name__=='__main__':main()
