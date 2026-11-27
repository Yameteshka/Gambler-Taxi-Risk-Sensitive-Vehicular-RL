"""Compare forecast state propagation with actual consecutive driving ticks."""
from pathlib import Path
from dataclasses import replace
import sys,json
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from env import resource_taxi as env
import navigation,protocol
c=env.Config(predictive_driver=True,predictive_escape=True,reward_shaping=False,loan_capacity=8)
params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),c)
_,starts=jax.vmap(env.ResourceTaxiEnv.reset,in_axes=(0,None))(jnp.arange(11,23),params)
starts=replace(starts,time=jnp.arange(149,161,dtype=jnp.int32),
    fuel=jnp.tile(jnp.array((0.,.001,12.)),4),fatigue=jnp.linspace(.1,.95,12),
    agent_pos=jnp.broadcast_to(jnp.array((24.,15.)),(12,2)))
@eqx.filter_jit
def compare(starts):
    def tick(carry,index):
        actual,forecast=carry
        controls=jnp.broadcast_to(jnp.array((.4,.15)),(12,2))
        _,actual,_,_,_=jax.vmap(env.ResourceTaxiEnv.step,in_axes=(0,0,0,None))(
            jax.random.split(jax.random.key(0),12),actual,env.Action(controls,jnp.zeros(12,jnp.int32)),params)
        def prediction(s,control):
            pos,direction,speed,steering=env.drive(s,control,params)
            traffic,velocity=env.traffic_step(s,params)
            return navigation.forecast_motion(s,params,pos,direction,speed,steering,traffic,velocity)
        forecast=jax.vmap(prediction)(forecast,controls)
        errors=jnp.array([jnp.max(jnp.abs(getattr(actual,k)-getattr(forecast,k))) for k in
            ('agent_pos','agent_forward_dir','speed','steering','fuel','fatigue','time','kinematic_obstacles','kinematic_obst_velocities')])
        return (actual,forecast),errors
    return jax.lax.scan(tick,(starts,starts),jnp.arange(12))
(actual,forecast),errors=compare(starts)
np.testing.assert_allclose(np.asarray(errors),0.,atol=3e-6,rtol=0.)
assert bool((actual.outcome==env.ACTIVE).all()),'Forecast comparison unexpectedly crossed a terminal state'
result={'passed':True,'cases':12,'consecutive_ticks':12,'maximum_absolute_error':float(errors.max()),
    'includes_day_boundary_empty_tank_fuel_exhaustion_and_high_fatigue':True,
    'source_hashes':protocol.source_hashes()}
(ROOT/'experiments/driver_forecast_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True)
