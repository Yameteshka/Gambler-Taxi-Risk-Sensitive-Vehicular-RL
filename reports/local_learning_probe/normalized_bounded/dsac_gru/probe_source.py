"""Fast synthetic ownership tasks using the production PPO / hybrid DSAC updates.

This deliberately abstracts travel and calendar, and is NOT real-game evidence.
Actions, voluntary loans, wealth accounting, survival and ownership are retained.
No action labels, behavior cloning, or evaluation-time action injection is used.
"""
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace
from dataclasses import replace
import argparse,json,sys,time,hashlib,shutil
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import models,strategy,protocol,resource_curriculum
import strategy_training as training
from algorithms import hybrid_dsac as dsac
from env import resource_taxi as env

COUNT=64
GAMMA=.97
CASES=('earning','food','fuel','fatigue','microloan','checkout')


class State(NamedTuple):
    cash:jax.Array
    debt:jax.Array
    micro:jax.Array
    food:jax.Array
    fuel:jax.Array
    tired:jax.Array
    clock:jax.Array
    paid:jax.Array
    case:jax.Array
    done:jax.Array
    won:jax.Array
    counters:jax.Array


PARAMS,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(public_food_menu=True))
BASE=strategy.reset(11,PARAMS)[0].features


def reset(key,size=COUNT,case=None):
    kind=jax.random.randint(key,(size,),0,6) if case is None else jnp.full(size,case)
    zeros=jnp.zeros(size)
    return State(jnp.where(kind==5,100.,jnp.where(kind==1,50.,jnp.where(kind==2,40.,0.))),
        jnp.where((kind==4)|(kind==5),0.,100.),jnp.where(kind==4,50.,0.),
        jnp.where(kind==1,.12,.9),jnp.where(kind==2,0.,5.),jnp.where(kind==3,.95,.1),
        jnp.zeros(size,dtype=jnp.int32),jnp.zeros(size,dtype=jnp.bool_),kind,
        jnp.zeros(size,dtype=jnp.bool_),jnp.zeros(size,dtype=jnp.bool_),jnp.zeros((size,6)))


def observation(s,unused=None):
    features=jnp.broadcast_to(BASE,(s.cash.size,BASE.size))
    features=features.at[:,68].set(s.cash/1000).at[:,69].set(s.debt/1000).at[:,70].set(s.micro/1000)
    features=features.at[:,73].set(s.fuel/25).at[:,74].set(s.food).at[:,75].set(s.tired)
    features=features.at[:,76].set(s.clock/32).at[:,77].set((s.clock%4)/4).at[:,81].set(s.paid)
    mask=jnp.ones((s.cash.size,strategy.COUNT),dtype=jnp.bool_)
    mask=mask.at[:,11:14].set(((s.debt>0)&(s.cash>0))[:,None])
    mask=mask.at[:,14:17].set(((s.micro>0)&(s.cash>0))[:,None])
    mask=mask.at[:,17:20].set(s.cash[:,None]>=jnp.array((10.,25.,55.)))
    mask=mask.at[:,20:23].set(s.cash[:,None]>=jnp.array((10.,30.,90.)))
    mask=mask.at[:,23:25].set(False)
    mask=mask.at[:,25].set((s.debt==0)&(s.micro==0))
    return SimpleNamespace(features=features,action_mask=mask)


def step(s,action,key):
    a=action.choice
    old_wealth=s.cash-s.debt-s.micro
    # Compressed travel: one synthetic step. Prices are drawn once per meal.
    u=jax.random.uniform(key,(s.cash.size,3))
    wage=150.*jnp.where(u[:,0]<.3,.75,jnp.where(u[:,0]<.7,1.,1.35))
    work=(a==1)&~s.paid&(s.fuel>=.25)
    venue=jnp.clip(a-3,0,2)
    price=jnp.array((5.,15.,35.))[venue]*(.8+.4*u[:,1])
    portion=jnp.array((.05,.15,.65))[venue]*(.8+.4*u[:,2])
    meal=(a>=3)&(a<=5)&(s.cash>=price)
    volume=jnp.minimum(25.-s.fuel,jnp.where(a==6,5.,25.))
    refill=((a==6)|(a==7))&(s.cash>=volume*1.2)
    borrow=(a>=8)&(a<=10)
    borrowed=jnp.where(borrow,jnp.array((25.,50.,100.))[jnp.clip(a-8,0,2)],0.)
    car=jnp.where((a>=11)&(a<=13),jnp.minimum(jnp.minimum(s.cash,s.debt),
        jnp.array((25.,100.,1e9))[jnp.clip(a-11,0,2)]),0.)
    micro=jnp.where((a>=14)&(a<=16),jnp.minimum(jnp.minimum(s.cash,s.micro),
        jnp.array((25.,100.,1e9))[jnp.clip(a-14,0,2)]),0.)
    wager=(a>=17)&(a<=22)
    bets=jnp.array((10.,25.,55.,10.,30.,90.))[jnp.clip(a-17,0,5)]
    # Blackjack is a Bernoulli surrogate here, not the production card engine.
    win=u[:,0]<jnp.where(a<20,.12,.48)
    gross=jnp.where(win,bets*jnp.where(a<20,325/55,1.9),0.)
    gambling=jnp.where(wager,gross-bets,0.)
    cash=s.cash+work*wage+borrowed-car-micro-meal*price-refill*volume*1.2+gambling
    debt=jnp.maximum(s.debt-car,0.)
    personal=jnp.maximum(s.micro+borrowed-micro,0.)
    food=jnp.clip(s.food-.035-.025*work+jnp.where(meal,portion,0.),0.,1.)
    fuel=jnp.clip(s.fuel-.25*work+jnp.where(refill,volume,0.),0.,25.)
    sleep=a==2
    tired=jnp.clip(s.tired+.025+.1*work-.8*sleep,0.,1.)
    # Fatigue is a controllable stochastic safety cost in this synthetic task.
    crash=work&(s.tired>.8)&(u[:,2]<.8)
    won=(a==25)&(debt==0)&(personal==0)&(food>0)&~crash
    done=won|crash|(food<=0)|(s.clock>=31)
    reward=won.astype(jnp.float32)-.1*(done&~won)-.1*(1-GAMMA)*~done
    reward+=.00005*((cash-debt-personal)-old_wealth)
    # Continuous style changes a small synthetic driving cost. It never labels a control.
    reward-=.00002*jnp.sum(action.control**2,axis=-1)*dsac.continuous_mask(strategy.COUNT)[a]
    events=jnp.stack((work,meal,refill,sleep,borrow,micro>0),axis=-1).astype(jnp.float32)
    new=State(cash,debt,personal,food,fuel,tired,s.clock+1,(s.paid|work)&((s.clock+1)%4!=0),
        s.case,done,won,s.counters+events)
    return new,reward,done


@eqx.filter_jit
def evaluate(actor,kind):
    def one(case):
        s=reset(jax.random.key(9000+case),128,case)
        hidden=models.initial_memory(kind,128,BASE.size)
        def tick(carry,index):
            s,hidden=carry
            obs=observation(s)
            action,hidden,*_=models.policy_step(actor,kind,obs.features,obs.action_mask,hidden,
                jax.random.fold_in(jax.random.key(999),index))
            new,_,_=step(s,action,jax.random.fold_in(jax.random.key(177),index))
            new=jax.tree.map(lambda x,y:jnp.where(s.done.reshape((128,)+(1,)*(x.ndim-1)),y,x),new,s)
            return (new,hidden),None
        (s,_),_=jax.lax.scan(tick,(s,hidden),jnp.arange(32))
        return s.won.mean(),s.counters.mean(axis=0)
    return jax.vmap(one)(jnp.arange(6))


def run(kind,settings,iterations,folder,exploration=0.,scope='Synthetic abstract economy, 128 sampled evaluation episodes per context; not real-game performance.'):
    key=jax.random.key(301)
    agent=training.dsac_agent(kind,dsac.AgentConfig(action_count=strategy.COUNT,gamma=GAMMA,**settings)) if kind.startswith('dsac') else None
    learner=agent.init(key,BASE.size) if agent else None
    actor=learner.actor if agent else models.init_ppo(key,BASE.size,kind=='ppo_gru',strategy.COUNT)
    optimizer=training.OPT
    opt_state=None if agent else optimizer.init(actor)
    params=SimpleNamespace(config=SimpleNamespace(exploration_epsilon=exploration,gamma=GAMMA))
    def one_observation(state,unused):
        batched=jax.tree.map(lambda value:value[None],state)
        obs=observation(batched)
        return env.Observation(jnp.float32(0.),jnp.zeros(2),jnp.zeros((16,2)),obs.features[0],obs.action_mask[0])
    ppo_update=training.make_ppo_update(params,optimizer,observation_fn=one_observation) if not agent else None
    s=reset(jax.random.key(11));hidden=models.initial_memory(kind,COUNT,BASE.size)
    update=training.make_dsac_update(agent,warmup=2048) if agent else None
    replay=None;cursor=jnp.int32(0);size=jnp.int32(0);history=[]
    @eqx.filter_jit
    def collect(actor,s,hidden,key):
        def tick(carry,index):
            s,hidden=carry
            obs=observation(s);old_hidden=hidden if kind.startswith('ppo') else jnp.zeros((COUNT,0))
            sequence_start=jnp.all(hidden==0,axis=tuple(range(1,hidden.ndim)))
            action,hidden,value,raw,logp,hc,hd=models.policy_step(actor,kind,obs.features,obs.action_mask,hidden,
                jax.random.fold_in(key,index),exploration_epsilon=exploration)
            new,reward,done=step(s,action,jax.random.fold_in(key,index+8))
            following=observation(new)
            t=dsac.Transition(obs.features,obs.action_mask,action.control,action.choice,reward,done,
                following.features,following.action_mask,jnp.full(COUNT,GAMMA),sequence_start)
            starts=reset(jax.random.fold_in(key,index+16))
            s=jax.tree.map(lambda a,b:jnp.where(done.reshape((COUNT,)+(1,)*(a.ndim-1)),b,a),new,starts)
            hidden=jnp.where(done.reshape((COUNT,)+(1,)*(hidden.ndim-1)),0.,hidden)
            return (s,hidden),(t,value,raw,logp,done,None,jnp.zeros(COUNT),old_hidden,hc,hd)
        return jax.lax.scan(tick,(s,hidden),jnp.arange(8))
    started=time.perf_counter()
    for iteration in range(iterations):
        key,collection_key,update_key=jax.random.split(key,3)
        (s,hidden),trace=collect(actor,s,hidden,collection_key)
        if agent:
            if replay is None:
                replay=jax.tree.map(lambda x:jnp.zeros((training.CAPACITY,*x.shape[2:]),x.dtype),trace[0])
            learner,replay,cursor,size,loss=update(learner,replay,cursor,size,trace[0],update_key);actor=learner.actor
        else:actor,opt_state,loss=ppo_update(actor,opt_state,trace,s,hidden,update_key)
        assert bool(jnp.isfinite(loss).all())
        if iteration%20==0 or iteration==iterations-1:
            rates,events=evaluate(actor,kind);rates=np.asarray(rates);events=np.asarray(events)
            row={'iteration':iteration,'transitions':(iteration+1)*512,'success_rates':dict(zip(CASES,rates.tolist())),
                 'mean_events':events.tolist(),'elapsed_seconds':time.perf_counter()-started}
            if agent:
                probe=observation(reset(jax.random.key(123)))
                features=probe.features
                if kind=='dsac_gru':
                    features=models.encode_history(learner.encoder,jnp.repeat(features[:,None],models.HISTORY,axis=1))
                critics=learner.inner.critics if kind=='dsac_gru' else learner.critics
                q=dsac._critics(critics,features,probe.action_mask,jnp.zeros((64,2)),agent.config.critic_layer_norm).mean(-1)
                row['q_min']=float(q.min());row['q_max']=float(q.max());row['losses']=np.asarray(loss).tolist()
            history.append(row);print(kind+' '+json.dumps(row),flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    stable=len(history)>=3 and all(min(row['success_rates'].values())>=.7 for row in history[-3:])
    result={'kind':kind,'settings':settings,'exploration':exploration,'history':history,'passed':stable,
        'acceptance':'Every context >=70% success in each of the final three checks; no best-checkpoint substitution.',
        'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':scope,
        'no_action_labels':True,'source_hashes':protocol.source_hashes()}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    shutil.copy2(Path(__file__),folder/'probe_source.py')
    models.save_policy(folder/'policy.npz',actor,{'kind':kind,'synthetic_only':True})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=160)
    parser.add_argument('--model',choices=('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru','both'),default='both')
    parser.add_argument('--variant',choices=('low_entropy','multihead','multihead_entropy','normalized','normalized_bounded','separate_memory'),default='low_entropy')
    parser.add_argument('--exploration',type=float,default=0.);args=parser.parse_args()
    settings={'actor_learning_rate':.0003,'learning_rate':.0003,'gradient_steps':64,'minibatch_size':128,
        'target_rate':.005,'automatic_temperature':False,'continuous_temperature':.0001,
        'discrete_temperature':.0001,'sample_target_action':True}
    if args.variant in ('multihead','multihead_entropy'):settings['multihead_critic']=True
    if args.variant=='multihead_entropy':settings['discrete_temperature']=.001
    if args.variant in ('normalized','normalized_bounded','separate_memory'):
        settings.update(critic_layer_norm=True,actor_learning_rate=.0001,discrete_temperature=.001)
    if args.variant in ('normalized_bounded','separate_memory'):
        settings.update(continuous_temperature=3e-6,discrete_temperature=3e-6)
    if args.variant=='separate_memory':settings['separate_recurrent_encoders']=True
    for kind in ('dsac_mlp','ppo_mlp') if args.model=='both' else (args.model,):
        result=run(kind,settings if kind.startswith('dsac') else {},args.iterations,ROOT/'reports/local_learning_probe'/args.variant/kind,args.exploration)
        print('RESULT '+kind+' '+str(result['passed']),flush=True)
