"""Four production agents after local learning checks, then frozen scenarios."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,os
import debt500_study as study
import ownership_curriculum as resource_curriculum

ROOT=Path(__file__).resolve().parents[1]
SETTINGS={'actor_learning_rate':.0001,'learning_rate':.0003,'gradient_steps':64,
    'minibatch_size':128,'target_rate':.005,'critic_layer_norm':True,
    'automatic_temperature':False,'continuous_temperature':3e-6,'discrete_temperature':3e-6,
    'sample_target_action':True,'multihead_critic':False,'separate_recurrent_encoders':True,
    'recurrent_observation_skip':True}


def configure():
    name=os.environ.get('GT_STUDY','resource_study')
    if name not in ('resource_study','resource_study_mlp','resource_study_gru'):raise ValueError('Unknown resource study')
    study.STUDY=ROOT/'artifacts'/name
    study.REPORTS=ROOT/'reports'/name
    study.CURRICULUM=resource_curriculum.NAME
    study.ORDER=(('hybrid_dsac','mlp'),('hybrid_ppo','mlp'),('hybrid_dsac','gru'),('hybrid_ppo','gru'))
    if name.endswith('_mlp'):study.ORDER=tuple(row for row in study.ORDER if row[1]=='mlp')
    if name.endswith('_gru'):study.ORDER=tuple(row for row in study.ORDER if row[1]=='gru')
    study.TRAINING_ONLY=name!='resource_study'
    study.ROW_EXTRAS={'reward_shaping':'disabled','navigation_cue':'disabled',
        'predictive_driver':'enabled','predictive_escape':'enabled','replay_warmup':'2048',
        'dsac_settings':json.dumps(SETTINGS,sort_keys=True),
        'network_decision_budget':'262144','physical_transition_budget':'33554432',
        'protocol_revision':'ownership_frontier_01'}
    study.STUDY_METADATA={'curriculum':{'name':resource_curriculum.NAME,
        'lessons':[asdict(lesson) for lesson in resource_curriculum.LESSONS],
        'original_principal':500,'transition_kernel_constant':True,'reset_rehearsal':.2,
        'min_physical_per_lesson':resource_curriculum.MIN_PHYSICAL,
        'min_network_decisions_per_lesson':resource_curriculum.MIN_DECISIONS,
        'diagnostic_cap_requires_both':True,
        'max_decisions_per_lesson':resource_curriculum.MAX_DECISIONS,
        'max_physical_per_lesson':resource_curriculum.MAX_PHYSICAL},
        'budget_per_model':33554432,'network_decision_budget_per_model':262144,
        'primary_training_budget_unit':'network decisions; physical ticks reported separately',
        'cold_start_fraction':None,'cold_exploration_epsilon':None,
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'external_objective_unchanged':True,'local_checks_required':True,
        'method':'Validated architectures trained sequentially on one ownership reset curriculum; frozen tests deferred for partial cohorts.',
        'agent_count':len(study.ORDER),'final_tests_deferred':study.TRAINING_ONLY,
        'driver_forecast':{'standard':40,'longer_guard':60,'candidate_holds':[4,12],
            'clock_fuel_fatigue_advance':True,
            'mean_route_distance_cost':.5,
            'planning_traffic_margin':0.,'actual_collision_radius_unchanged':True,
            'short_guard_fallback_when_all_nominal_continuations_fail':True}}


if __name__=='__main__':
    configure()
    try:
        study.main()
        from resource_results import write_results
        write_results(ROOT,study.STUDY.name)
    except BaseException as error:
        if study.STUDY.exists():study.status(state='stopped',error=repr(error))
        raise
