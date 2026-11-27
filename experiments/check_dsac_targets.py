"""Distributional target regression checks against exact categorical enumeration."""
from pathlib import Path
from dataclasses import replace
import sys,json
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from algorithms import hybrid_dsac as d
config=d.AgentConfig(action_count=26,quantiles=8,automatic_temperature=False,
                    continuous_temperature=3e-6,discrete_temperature=3e-6,
                    critic_layer_norm='--normalized' in sys.argv,multihead_critic='--multihead' in sys.argv)
agent=d.HybridDSAC(config);learner=agent.init(jax.random.key(1),194)
x=jax.random.normal(jax.random.key(2),(32,194))*.2
mask=jnp.ones((32,26),dtype=jnp.bool_).at[:,10:13].set(False)
batch=d.Transition(x,mask,jnp.zeros((32,2)),jnp.zeros(32,dtype=jnp.int32),
    jnp.linspace(-.1,1.,32),jnp.arange(32)%3==0,x*.9,mask,jnp.linspace(.8,.999,32))
key=jax.random.key(9)
target,prob=d.bootstrap_distribution(learner,key,batch,config)
sampled,_=d.bootstrap_distribution(learner,key,batch,replace(config,sample_target_action=True))
control_key,choice_key=jax.random.split(key)
_,_,_,logp=d._sample_all(learner.actor,control_key,batch.next_features,batch.next_mask)
choices=jax.random.categorical(choice_key,logp)
assert not np.asarray((choices>=10)&(choices<13)).any()
expected=jnp.take_along_axis(target,choices[:,None,None],axis=1)
np.testing.assert_allclose(sampled,expected,atol=2e-6,rtol=3e-6)
terminal=np.asarray(batch.done)
np.testing.assert_allclose(np.asarray(sampled)[terminal,0],np.repeat(np.asarray(batch.reward)[terminal,None],8,axis=1))
pred=d._taken(learner.critics,batch)
full=d._distribution_loss(pred,target,prob,config)
parts=jnp.stack([d._distribution_loss(pred,target,jnp.broadcast_to(jax.nn.one_hot(i,26),(32,26)),config)
                for i in range(26)])
# A sample-specific probability requires retaining the batch axis; the strongest
# loss check uses a common probability vector with the same exact targets.
p=jax.nn.softmax(jnp.linspace(-1.,1.,26));p=p.at[10:13].set(0);p=p/p.sum()
mixture=d._distribution_loss(pred,target,jnp.broadcast_to(p,(32,26)),config)
np.testing.assert_allclose(mixture,(parts*p).sum(),rtol=2e-6)
new,metrics=jax.jit(agent.update)(learner,jax.random.key(10),batch)
assert np.isfinite(np.asarray(metrics)).all()
np.testing.assert_array_equal(new.log_temperatures,learner.log_temperatures)
result={'passed':True,'masked_sampling_matches_enumeration':True,'terminal_no_bootstrap':True,
    'mixture_loss_equals_weighted_component_losses':True,'fixed_temperatures_unchanged':True,
    'critic_layer_norm':config.critic_layer_norm,'multihead':config.multihead_critic}
from strategy_training import with_history
values=jnp.arange(16,dtype=jnp.float32)[:,None]
tiny=d.Transition(values,jnp.ones((16,26),dtype=jnp.bool_),jnp.zeros((16,2)),jnp.zeros(16,dtype=jnp.int32),
    jnp.zeros(16),jnp.zeros(16,dtype=jnp.bool_),values+1,jnp.ones((16,26),dtype=jnp.bool_),
    jnp.ones(16)*.99,jnp.zeros(16,dtype=jnp.bool_).at[6].set(True))
batch_one=jax.tree.map(lambda x:x[jnp.array((8,))],tiny)
history=with_history(tiny,batch_one,jnp.array((8,)),lanes=2,history=4)
np.testing.assert_array_equal(history.features[0,:,0],jnp.array((0.,0.,6.,8.)))
result['history_does_not_cross_reset_only_curriculum_boundary']=True
suffix=('_normalized' if config.critic_layer_norm else '')+('_multihead' if config.multihead_critic else '')
(ROOT/f'experiments/dsac_target_verification{suffix}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
