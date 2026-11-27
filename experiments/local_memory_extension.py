"""Prospectively extend the slow recurrent calibration, retaining the 600-step run."""
from pathlib import Path
import local_learning_probe as probe

settings={'actor_learning_rate':.0001,'learning_rate':.0003,'gradient_steps':64,
    'minibatch_size':128,'target_rate':.005,'automatic_temperature':False,
    'continuous_temperature':3e-6,'discrete_temperature':3e-6,
    'sample_target_action':True,'critic_layer_norm':True,
    'separate_recurrent_encoders':True,'recurrent_observation_skip':True}
result=probe.run('dsac_gru',settings,800,
    probe.ROOT/'reports/local_learning_probe/separate_memory_skip_extended/dsac_gru',.2)
print('EXTENDED_PROBE_RESULT',result['passed'],flush=True)
if not result['passed']:raise SystemExit(2)
