"""Every directed venue pair under matched headings and clock phases.

Resource depletion is isolated from routing. These are engineering development
cases, not held-out economic-policy evaluations.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import navigation
from env import resource_taxi as env,city
parser=argparse.ArgumentParser();parser.add_argument('--escape',action='store_true');args=parser.parse_args()
cases=[(a,b,heading,clock) for a in range(10) for b in range(10) if a!=b
    for heading in (0.,float(np.pi)) for clock in (0,10,80)]
start,goal,heading,clock=map(jnp.asarray,zip(*cases));n=len(cases)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),env.Config(days=0,loan_capacity=8,
    hunger_tick=0.,hunger_drive=0.,hunger_work=0.,hunger_push=0.,
    predictive_driver=True,predictive_escape=args.escape,reward_shaping=False))
_,states=jax.vmap(env.ResourceTaxiEnv.reset,in_axes=(0,None))(jnp.arange(11,11+n),params)
states=replace(states,agent_pos=params.zone_centers[start],fuel=jnp.full(n,25.),
    time=clock.astype(jnp.int32),agent_forward_dir=jnp.stack((jnp.cos(heading),jnp.sin(heading)),axis=1))
def set_traffic_phase(s,phase):
    def tick(i,s):
        positions,velocity=env.traffic_step(s,params)
        return replace(s,kinematic_obstacles=positions,kinematic_obst_velocities=velocity)
    return jax.lax.fori_loop(0,phase,tick,s)
states=jax.vmap(set_traffic_phase)(states,clock.astype(jnp.int32))
@eqx.filter_jit
def run(states):
    def tick(carry,index):
        s,arrived,arrival=carry
        controls=jax.vmap(lambda one,z:navigation.predictive_guard(one,params,z,.9,1.))(s,goal)
        _,new,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
            jax.random.split(jax.random.key(0),n),s,env.Action(controls,jnp.zeros(n,dtype=jnp.int32)),params)
        reached=(jnp.linalg.norm(new.agent_pos-params.zone_centers[goal],axis=-1)<=params.config.zone_radius)&(new.speed<params.config.parking_speed)&(new.outcome==env.ACTIVE)
        freeze=arrived|(s.outcome!=env.ACTIVE)
        new=jax.tree.map(lambda a,b:jnp.where(freeze.reshape((n,)+(1,)*(a.ndim-1)),b,a),new,s)
        arrival=jnp.where(reached&~arrived,index+1,arrival)
        return (new,arrived|reached,arrival),None
    return jax.lax.scan(tick,(states,jnp.zeros(n,bool),jnp.zeros(n,jnp.int32)),jnp.arange(384))[0]
started=time.perf_counter();end,arrived,arrival=run(states);jax.block_until_ready(arrived)
ok=np.asarray(arrived);crashed=np.asarray(end.outcome)==env.CRASH
result={'escape':args.escape,'routes':n,'arrivals':int(ok.sum()),'collisions':int(crashed.sum()),
    'nonterminal_origin_routes':int((np.asarray(start)!=city.GARAGE).sum()),
    'nonterminal_origin_arrivals':int(ok[np.asarray(start)!=city.GARAGE].sum()),
    'nonterminal_origin_collisions':int(crashed[np.asarray(start)!=city.GARAGE].sum()),
    'unfinished':int((~ok&~crashed).sum()),'elapsed_seconds':time.perf_counter()-started,
    'failed_cases':[{'from':city.NAMES[a],'to':city.NAMES[b],'heading':h,'clock':t,
        'collision':bool(crashed[i]),'position':np.asarray(end.agent_pos[i]).tolist(),
        'speed':float(end.speed[i])} for i,(a,b,h,t) in enumerate(cases) if not ok[i]],
    'arrival_tick_percentiles':np.percentile(np.asarray(arrival)[ok],(50,90,100)).tolist() if ok.any() else [],
    'scope':'All 90 directed venue pairs, two headings, three clock phases with advanced traffic; food depletion disabled, initial fuel 25. Fixed driving style. Garage is terminal-only during ordinary policy execution. Engineering development diagnostic only.'}
p=ROOT/f'reports/local_learning_probe/all_routes_clock_escape_{args.escape}.json';p.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='failed_cases'}),flush=True)
