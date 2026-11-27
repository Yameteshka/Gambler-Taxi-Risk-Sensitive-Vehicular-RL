"""Counterexample search for food-context feasibility, never training labels.

These manually chosen actions are a simulator diagnostic. They are not supplied
to an actor, replay buffer, checkpoint, curriculum gate, or final evaluation.
"""
from pathlib import Path
from dataclasses import replace
import sys,json,argparse
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env
import strategy,resource_curriculum
parser=argparse.ArgumentParser();parser.add_argument('--start-time',type=int,default=0)
parser.add_argument('--cleared',action='store_true')
parser.add_argument('--escape',action='store_true')
args=parser.parse_args()

c=replace(resource_curriculum.configuration(env.Config(starting_debt=500.,
    public_food_menu=True,reward_shaping=False,navigation_cue=False,predictive_driver=True),3),
    training_rehearsal=0.,predictive_escape=args.escape)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),c)
ids=jnp.arange(11,27)
_,states=jax.vmap(strategy.reset,in_axes=(0,None))(ids,params)
states=replace(states,time=jnp.full(16,args.start_time,dtype=jnp.int32))
if args.cleared:states=replace(states,car_debt=jnp.zeros(16),car_paid=jnp.full(16,500.))
advance=eqx.filter_jit(jax.vmap(lambda s,a:strategy.advance(s,a,params)))
rows=[]
sequence=(strategy.RESTAURANT,strategy.BUY,strategy.BUY) if args.cleared else (strategy.RESTAURANT,strategy.WORK,strategy.WORK,strategy.CAR_ALL,strategy.BUY)
for option in sequence:
    _,states,_,_,trace=advance(states,env.Action(jnp.zeros((16,2)),jnp.full(16,option,dtype=jnp.int32)))
    row={'operation':strategy.NAMES[option],
        **{name:np.asarray(getattr(states,name)).tolist() for name in
           ('time','cash','car_debt','satiety','outcome','agent_pos','speed')},
        'meals':np.asarray(states.counters[:,env.MEALS]).tolist(),
        'paid_shifts':np.asarray(states.counters[:,env.PAID]).tolist()}
    rows.append(row);print(json.dumps(row),flush=True)
result={'scope':'Manually chosen diagnostic sequence, not learned performance or training data.',
    'start_time':args.start_time,
    'scenario_ids':np.asarray(ids).tolist(),'sequence':rows,
    'ownership':int((np.asarray(states.outcome)==env.SUCCESS).sum())}
p=ROOT/f'reports/local_learning_probe/real_food_path_time_{args.start_time}_cleared_{args.cleared}_escape_{args.escape}.json';p.write_text(json.dumps(result,indent=2))
