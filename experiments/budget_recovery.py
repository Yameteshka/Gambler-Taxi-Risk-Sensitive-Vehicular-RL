"""Register recovery trials or run the cluster cohort and its frozen scenarios.

Defaults preserve the previous feedback duration and entropy settings, isolating
the new curriculum. Other treatments receive distinct output directories.
Only cluster cohort mode evaluates final scenarios after readiness.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,os,hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import jax,numpy as np
import protocol,models,budget_curriculum as curriculum
from env import resource_taxi as env
from strategy_training import train
from evaluate import evaluate
from resource_study import SETTINGS

COHORT='budget_recovery_cohort'
TUNED_COHORT='budget_recovery_tuned'
REFERENCE_SHA256='9b2a150d1543cf4a4728ee792175c73480a4b0ba11019f8f92279852acb1409f'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--wait-ticks',type=int,choices=(1,8),default=1)
    parser.add_argument('--alpha-d',type=float,choices=(3e-6,3e-5,3e-4),default=3e-6)
    parser.add_argument('--actor-update',choices=('reverse_kl','forward_kl'),default='reverse_kl')
    parser.add_argument('--ppo-native-exploration',action='store_true')
    parser.add_argument('--cluster-cohort',action='store_true')
    parser.add_argument('--tuned-cohort',action='store_true')
    parser.add_argument('--ppo-decisions',type=int,default=262144)
    parser.add_argument('--standard-rehearsal',type=float,default=0.)
    parser.add_argument('--dsac-reverse-after-fraction',type=float,default=0.)
    parser.add_argument('--readiness-inference',choices=('sample','argmax_mean'),default='sample')
    parser.add_argument('--model',choices=('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru'),action='append')
    args=parser.parse_args()
    if args.cluster_cohort and args.tuned_cohort:
        parser.error('Choose one cohort')
    cohort=args.cluster_cohort or args.tuned_cohort
    if args.ppo_decisions<262144 or args.ppo_decisions%512:
        parser.error('PPO budget must be at least 262144 and divisible by 512')
    if not 0.<=args.standard_rehearsal<=1.:
        parser.error('Standard rehearsal must be a probability')
    if args.dsac_reverse_after_fraction and not (0.<args.dsac_reverse_after_fraction<1. and args.actor_update=='forward_kl'):
        parser.error('A switch requires forward KL and a fraction in (0, 1)')
    followup=(args.ppo_decisions!=262144 or args.standard_rehearsal or
              args.dsac_reverse_after_fraction or args.readiness_inference!='sample' or args.model)
    if followup and cohort:
        parser.error('Followup settings require a separate study')
    identity=f'food_budget_wait{args.wait_ticks}_alpha{args.alpha_d:g}'
    if args.actor_update!='reverse_kl':identity+='_'+args.actor_update
    if args.ppo_native_exploration:identity+='_ppo_native'
    identity+='_r2'
    if followup:
        identity+=f'_ppo{args.ppo_decisions}_home{args.standard_rehearsal:g}_switch{args.dsac_reverse_after_fraction:g}_{args.readiness_inference}'
    if cohort:
        identity=TUNED_COHORT if args.tuned_cohort else COHORT
        args.actor_update='forward_kl'
        args.ppo_native_exploration=True
    if args.tuned_cohort:
        args.standard_rehearsal=.25
        args.readiness_inference='argmax_mean'
    output=ROOT/'artifacts'/identity
    report=ROOT/'reports'/identity;report.mkdir(parents=True,exist_ok=True)
    settings={**SETTINGS,'discrete_temperature':args.alpha_d,'categorical_actor_update':args.actor_update}
    kinds=('dsac_mlp','ppo_mlp','ppo_gru') if cohort else ('dsac_mlp','ppo_mlp','dsac_gru','ppo_gru')
    if args.model:
        kinds=tuple(kind for kind in kinds if kind in args.model)
        identity+='_'+'-'.join(kinds)
    baseline=protocol.registry()[0]
    rows=[]
    for kind in kinds:
        decisions=args.ppo_decisions if kind.startswith('ppo') else 262144
        tuned_ppo=args.tuned_cohort and kind.startswith('ppo')
        rows.append({**baseline,'run_id':kind,
            'algorithm':'hybrid_dsac' if kind.startswith('dsac') else 'hybrid_ppo',
            'architecture':kind.split('_')[1],'initialization_seed':'301',
            'curriculum':curriculum.NAME,'starting_debt':'500','training_horizon':'10d',
            'exploration':'standard','training_market':'dynamic','casino':'enabled',
            'public_food_menu':'enabled','reward_shaping':'disabled','navigation_cue':'disabled',
            'predictive_driver':'enabled','predictive_escape':'enabled',
            'ppo_native_exploration':'enabled' if args.ppo_native_exploration and kind.startswith('ppo') else 'disabled',
            'unavailable_wait_ticks':str(args.wait_ticks),
            'network_decision_budget':str(decisions),'physical_transition_budget':str(decisions*128),
            'lesson_decision_cap':str(max(curriculum.MAX_DECISIONS,decisions//4)),
            'standard_start_rehearsal':str(args.standard_rehearsal),
            'dsac_reverse_after_fraction':str(args.dsac_reverse_after_fraction if kind.startswith('dsac') else 0.),
            'readiness_inference':args.readiness_inference,
            'dsac_reverse_at_stage':'12' if args.tuned_cohort and kind=='dsac_mlp' else '-1',
            'late_reserve_decisions':'102400' if args.tuned_cohort else '0',
            'resource_observation':'enabled' if args.tuned_cohort else 'disabled',
            'observation_history':'enabled' if args.tuned_cohort and kind.endswith('_mlp') else 'disabled',
            'ppo_epochs':'12' if tuned_ppo else '3',
            'ppo_target_kl':'0.02' if tuned_ppo else '0',
            'ppo_clip':'0.15' if tuned_ppo else '0.2',
            'ppo_value_head_lr_multiplier':'3' if tuned_ppo and kind=='ppo_gru' else '1',
            'ppo_normalize_value_loss':'enabled' if tuned_ppo else 'disabled',
            'ppo_exploration_decay':'enabled' if tuned_ppo else 'disabled',
            'dsac_settings':json.dumps(settings,sort_keys=True),'replay_warmup':'2048',
            'protocol_revision':'food_budget_tuned_03' if args.tuned_cohort else 'food_budget_retention_02','status':'planned'})
    plan={'identity':identity,'rows':rows,'lessons':[x.name for x in curriculum.LESSONS],
        'development_banks':[list(curriculum.VALIDATION_IDS),list(curriculum.SECOND_BANK)],
        'required_ownership_per_16':12,'selected_standard_required_ownership_per_64':curriculum.STANDARD_OWNERSHIP_MIN,
        'matched_actor_checks':'Current context on two banks plus earlier essential contexts on both banks.',
        'final_evaluations_enabled':cohort,'source_hashes':protocol.source_hashes(),
        'status':'Recovery protocol registered.',
        'single_initialization_limitation':True}
    if cohort:
        plan['reference']={'model':'dsac_gru','sha256':REFERENCE_SHA256,'trained_with':'original ownership frontier and reverse KL'}
    (report/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if not args.train:
        print('RECOVERY_PLAN',report/'plan.json',flush=True);return
    checks=ROOT/'reports/resource_study_review_2026-11-23/recovery_checks/checks.json'
    if args.tuned_cohort:
        checks=ROOT/'reports/local_food_recovery/tuned_preflight.json'
    if not checks.exists() or not json.loads(checks.read_text())['passed']:
        raise RuntimeError('Recovery engineering checks must pass before training')
    if json.loads(checks.read_text()).get('source_hashes')!=protocol.source_hashes():
        raise RuntimeError('Recovery source changed after engineering checks')
    output.mkdir(parents=True,exist_ok=True)
    freeze=output/'freeze.json'
    if freeze.exists() and json.loads(freeze.read_text())!=plan:
        raise RuntimeError('Recovery freeze differs; register a separate treatment')
    if not freeze.exists():freeze.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[]
    for row in rows:
        kind=row['run_id'];folder=output/'training'/kind
        try:
            if not (folder/'complete.json').exists():
                train(row,resume=(folder/'resume.pkl.gz').exists(),output_root=(output/'training').relative_to(ROOT))
        except RuntimeError:
            failure=folder/'curriculum_failure.json'
            if not failure.exists():raise
            results.append({'model':kind,'qualified':False,'failure':json.loads(failure.read_text())})
            (report/'readiness.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            continue
        actor,metadata=models.load_policy(folder/'selected.npz')
        config=env.Config(**metadata['evaluation_environment'])
        params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
        standard,_,_=evaluate(actor,kind,params,range(1000001,1000065),
            output/'readiness'/kind/'standard',deterministic=args.readiness_inference=='argmax_mean',interface='strategy')
        qualified=standard['ownership']>=curriculum.STANDARD_OWNERSHIP_MIN;retained=[]
        for name,stage,context,ids in curriculum.additional_checks(config,len(curriculum.LESSONS)-1):
            summary,states,_=evaluate(actor,kind,replace(params,config=context),ids,
                output/'readiness'/kind/name,deterministic=args.readiness_inference=='argmax_mean',interface='strategy')
            success=np.asarray(states.outcome)==env.SUCCESS;counts=np.asarray(states.counters)
            result=curriculum.gate(stage,summary,int((success&(counts[:,env.PAID]>0)).sum()),True,
                successful_meal_episodes=int((success&(counts[:,env.MEALS]>0)).sum()),
                successful_refuel_episodes=int((success&(counts[:,env.REFILLS]>0)).sum()))
            qualified &= result['passed'];retained.append({'name':name,**result})
        results.append({'model':kind,'qualified':bool(qualified),'standard':standard,'retained':retained})
        (report/'readiness.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if cohort:
        run_frozen_cohort(output,report,results,deterministic=args.tuned_cohort)
    print('DEVELOPMENT_RECOVERY_FINISHED',json.dumps(results),flush=True)


def run_frozen_cohort(output,report,results,deterministic=False):
    from debt500_study import variants
    from audit_results import audit

    reference=ROOT/'experiments/recovery_reference.npz'
    if hashlib.sha256(reference.read_bytes()).hexdigest()!=REFERENCE_SHA256:
        raise RuntimeError('Frozen DSAC GRU checkpoint checksum mismatch')
    actor,metadata=models.load_policy(reference)
    config=env.Config(**metadata['evaluation_environment'])
    params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),config)
    standard,_,_=evaluate(actor,'dsac_gru',params,range(1000001,1000065),
        output/'readiness/dsac_gru/standard',deterministic=deterministic,interface='strategy')
    results.append({'model':'dsac_gru','qualified':standard['ownership']>=curriculum.STANDARD_OWNERSHIP_MIN,
        'standard':standard,'frozen_reference':True,'sha256':REFERENCE_SHA256})
    (report/'readiness.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if len(results)!=4 or not all(row['qualified'] for row in results):
        state='needs_learning_diagnosis'
        (output/'status.json').write_text(json.dumps({'state':state,'final_evaluation_record_count':0}),encoding='utf-8')
        print('COHORT_GATE',state,flush=True)
        return
    completed=0
    for row in results:
        kind=row['model']
        checkpoint=reference if kind=='dsac_gru' else output/'training'/kind/'selected.npz'
        actor,metadata=models.load_policy(checkpoint)
        config=env.Config(**metadata['evaluation_environment'])
        scenarios=list(variants(config))
        if len(scenarios)!=30 or len({label for label,_,_,_ in scenarios})!=30:
            raise RuntimeError('Expected 30 unique frozen conditions')
        banks={'test_standard':3000001}
        for label,changed,bank,controller in scenarios:
            use_argmax=deterministic or label=='deterministic'
            if deterministic and label=='deterministic':
                # The standard condition already uses argmax. Preserve the inference ablation.
                label='sampled_policy'
                use_argmax=False
            if bank not in banks:banks[bank]=3100001+100000*(len(banks)-1)
            path=output/'evaluation'/kind/label
            params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0),changed)
            if not (path/'audit.json').exists():
                evaluate(actor,kind,params,range(banks[bank],banks[bank]+256),path,
                    deterministic=use_argmax,controller=controller,interface='strategy')
                audit(path,checkpoint)
            verified=json.loads((path/'audit.json').read_text())
            if not verified['passed'] or not verified['policy_reproduced']:
                raise RuntimeError(f'Audit failed for {kind}/{label}')
            completed+=1
            (output/'status.json').write_text(json.dumps({'state':'evaluating','model':kind,'condition':label,
                'final_evaluation_record_count':completed}),encoding='utf-8')
    (output/'status.json').write_text(json.dumps({'state':'evaluations_complete','final_evaluation_record_count':completed}),encoding='utf-8')


if __name__=='__main__':main()
