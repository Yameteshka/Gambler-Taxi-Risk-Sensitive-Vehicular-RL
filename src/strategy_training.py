"""Service-level RL under shared driving, with physical-step accounting and full resume."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import gzip
import shutil
from time import perf_counter
from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms import hybrid_dsac as dsac
from algorithms.recurrent_dsac import RecurrentDSAC
from env import resource_taxi as env
from evaluate import evaluate
import models
import protocol
import progressive_curriculum as gradual
import resource_curriculum
import ownership_curriculum
import budget_curriculum
from telemetry import frame, write_chunk
import strategy
from evaluate_strategy import flatten_physical


ENVS, ROLLOUT, CAPACITY = 64, 8, 65536
OPT = optax.chain(optax.clip_by_global_norm(.5), optax.adam(2.5e-4))


def behavior_epsilon(config, kind):
    return 0. if kind.startswith("ppo") and config.ppo_native_exploration else config.exploration_epsilon


@lru_cache(maxsize=64)
def compiled_collector(config, kind, scenario_count):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    design = {**protocol.load(), "training_scenario_count": scenario_count}
    return make_collector(params, kind, design)


@lru_cache(maxsize=32)
def compiled_ppo(config, diagnostics=False):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), config)
    return make_ppo_update(params, OPT, diagnostics=diagnostics)


@lru_cache(maxsize=16)
def dsac_agent(kind, config):
    return RecurrentDSAC(config) if kind == "dsac_gru" else dsac.HybridDSAC(config)


def reset_batch(params, design, counters):
    first, last = protocol.check_splits(design)
    ids = first+(counters % (last-first+1))
    return jax.vmap(strategy.reset, in_axes=(0, None))(ids, params)


def cold_coverage(output):
    records = [json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()]
    records = [record for record in records if record['stage'] == 0]
    names = ('paid', 'meals', 'sleeps', 'refills', 'spins', 'hands', 'loans', 'car_payments', 'micro_payments')
    totals = {name: sum(r.get('completed_actions', {}).get(name, 0) for r in records) for name in names}
    result = {'training_only': True, 'completed_actions': totals,
        **{name: sum(r.get(name, 0) for r in records) for name in
           ('ownership_events', 'car_loan_closures', 'microloan_closures')}}
    result['passed'] = all(totals.values()) and all(result[name] > 0 for name in
        ('ownership_events', 'car_loan_closures', 'microloan_closures'))
    (output/'cold_start_coverage.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def make_collector(params, kind, design):
    @eqx.filter_jit
    def collect(actor, states, hidden, episode_counters, rng, intrinsic):
        def tick(carry, index):
            states, hidden, counters, rng = carry
            rng, action_key = jax.random.split(rng)
            obs = jax.vmap(strategy.get_observation, in_axes=(0, None))(states, params)
            initial_hidden = hidden if kind.startswith("ppo") else jnp.zeros((ENVS, 0))
            sequence_start=jnp.all(hidden==0,axis=tuple(range(1,hidden.ndim)))
            actions, hidden, value, raw, logp, entropy_c, entropy_d = models.policy_step(
                actor, kind, obs.features, obs.action_mask, hidden, action_key,
                exploration_epsilon=behavior_epsilon(params.config,kind))
            next_obs, new, reward, done, telemetry = jax.vmap(strategy.advance, in_axes=(0,0,None))(states, actions, params)
            duration = telemetry["active"].sum(axis=1)
            discount = params.config.gamma**duration
            telemetry["entropy_c"] = jnp.broadcast_to(entropy_c[:,None], (ENVS,strategy.STEPS))
            telemetry["entropy_d"] = jnp.broadcast_to(entropy_d[:,None], (ENVS,strategy.STEPS))
            telemetry["raw_style"] = jnp.broadcast_to(raw[:,None], (ENVS,strategy.STEPS,2))
            bonus = jnp.zeros(ENVS)
            if intrinsic is not None:
                x = jnp.clip(obs.features[:,:16], -1., 1.)
                target = dsac._mlp(intrinsic["target"],x)
                guess = dsac._mlp(intrinsic["predictor"],x)
                bonus = .05*(1-discount)*jnp.clip(((target-guess)**2).mean(-1)/.01,0.,1.)*intrinsic["weight"]
            telemetry["intrinsic_reward"] = bonus[:,None]*(jnp.arange(strategy.STEPS)==0)[None]
            transition = dsac.Transition(obs.features,obs.action_mask,actions.control,actions.choice,
                                         reward+bonus,done,next_obs.features,next_obs.action_mask,discount,sequence_start)
            counters += ENVS*done
            _, starts = reset_batch(params,design,counters)
            states = jax.tree.map(lambda a,b:jnp.where(done.reshape((ENVS,)+(1,)*(a.ndim-1)),b,a),new,starts)
            hidden = jnp.where(done.reshape((ENVS,)+(1,)*(hidden.ndim-1)),0.,hidden)
            return (states,hidden,counters,rng),(transition,value,raw,logp,done,telemetry,bonus,initial_hidden,entropy_c,entropy_d)
        return jax.lax.scan(tick,(states,hidden,episode_counters,rng),jnp.arange(ROLLOUT))
    return collect


def make_ppo_update(params, opt, observation_fn=strategy.get_observation, diagnostics=False):
    config=params.config
    epsilon=behavior_epsilon(config,"ppo")
    @eqx.filter_jit
    def update(model, opt_state, trace, end_states, end_hidden, rng):
        transitions, values, raw, old_logp, done, _, bonus, old_hidden, _, _ = trace
        obs = jax.vmap(observation_fn, in_axes=(0, None))(end_states, params)
        _, _, _, next_value, _ = models.ppo_step(model, obs.features, obs.action_mask, end_hidden)
        following = jnp.concatenate((values[1:], next_value[None]))
        resets = jnp.concatenate((jnp.zeros((1, ENVS), dtype=jnp.bool_), done[:-1]))

        def gae(carry, row):
            reward, terminal, value, next_value, discount = row
            continuation = discount*~terminal
            carry = reward+continuation*next_value-value+continuation*.95*carry
            return carry, carry
        _, advantage = jax.lax.scan(gae, jnp.zeros(ENVS),
            (transitions.reward, done, values, following, transitions.discount), reverse=True)
        returns = advantage+values
        return_variance=returns.var()
        explained_variance=jnp.where(return_variance>1e-8,
            1.-(returns-values).var()/jnp.maximum(return_variance,1e-8),0.)
        value_scale=jnp.maximum(returns.std(),.25) if config.ppo_normalize_value_loss else 1.
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-6)

        def minibatch(carry, indices):
            model, opt_state, stopped, applied = carry
            entropy_noise = jax.random.normal(jax.random.fold_in(rng, indices[0]+731), raw[:, indices].shape)

            def loss(candidate):
                def forward(hidden, row):
                    features, masks, reset = row
                    hidden = jnp.where(reset[:, None], 0., hidden)
                    mean, std, logp, value, hidden = models.ppo_step(candidate, features, masks, hidden,epsilon)
                    return hidden, (mean, std, logp, value)
                _, (mean, std, logp, prediction) = jax.lax.scan(
                    forward, old_hidden[0, indices],
                    (transitions.features[:, indices], transitions.mask[:, indices], resets[:, indices]))
                lp, _, _ = models.hybrid_logprob(
                    mean, std, logp, raw[:, indices], transitions.choice[:, indices])
                entropy_c, entropy_d = models.hybrid_entropy(mean, std, logp, entropy_noise)
                log_ratio=jnp.clip(lp-old_logp[:,indices],-20.,20.)
                ratio=jnp.exp(log_ratio)
                objective = jnp.minimum(ratio*advantage[:, indices],
                                         jnp.clip(ratio,1-config.ppo_clip,1+config.ppo_clip)*advantage[:, indices])
                value_loss=jnp.mean(((prediction-returns[:,indices])/value_scale)**2)
                entropy = .003*entropy_c.mean()+.01*entropy_d.mean()
                exploration_loss=0.
                if config.ppo_native_exploration and config.exploration_epsilon>0.:
                    legal=transitions.mask[:,indices]
                    uniform_cross_entropy=-jnp.where(legal,logp,0.).sum(-1)/legal.sum(-1)
                    exploration_loss=.01*config.exploration_epsilon*uniform_cross_entropy.mean()
                approx_kl=jnp.mean(jnp.expm1(log_ratio)-log_ratio)
                return -objective.mean()+.5*value_loss-entropy+exploration_loss,approx_kl

            def attempt(_):
                (value,approx_kl),grad=jax.value_and_grad(loss,has_aux=True)(model)
                reject=(approx_kl>config.ppo_target_kl) if config.ppo_target_kl>0. else jnp.bool_(False)

                def apply(_):
                    changes,new_state=opt.update(grad,opt_state)
                    if config.ppo_value_head_lr_multiplier!=1.:
                        w,b=changes['head'][-1]
                        value_head=(w.at[:,-1].multiply(config.ppo_value_head_lr_multiplier),
                                    b.at[-1].multiply(config.ppo_value_head_lr_multiplier))
                        changes={**changes,'head':(*changes['head'][:-1],value_head)}
                    return optax.apply_updates(model,changes),new_state

                candidate,new_state=jax.lax.cond(reject,lambda _:(model,opt_state),apply,operand=None)
                metrics=jnp.stack((jnp.where(reject,0.,value),approx_kl))
                return (candidate,new_state,reject,applied+(~reject)),metrics

            return jax.lax.cond(stopped,lambda _:(carry,jnp.zeros(2,dtype=values.dtype)),attempt,operand=None)

        groups = jax.vmap(lambda k: jax.random.permutation(k, ENVS))(
            jax.random.split(rng, config.ppo_epochs)).reshape((-1, 8))
        (model,opt_state,_,applied),metrics=jax.lax.scan(minibatch,
            (model,opt_state,jnp.bool_(False),jnp.int32(0)),groups)
        loss=metrics[:,0].sum()/jnp.maximum(applied,1)
        result=jnp.stack((loss,applied,metrics[:,1].max(),explained_variance)) if diagnostics else loss
        return model,opt_state,result
    return update


def with_history(replay, batch, indices, lanes=ENVS, history=models.HISTORY):
    """Gather same-lane sequences and zero everything before the latest reset."""
    backward = (indices[:, None]-lanes*jnp.arange(history)[None]) % replay.features.shape[0]
    values = replay.features[backward]
    boundaries=replay.done[backward[:,1:]]
    if replay.sequence_start is not None:
        boundaries=boundaries|replay.sequence_start[backward[:,:-1]]
    past_terminal = jnp.cumsum(boundaries, axis=1) > 0
    valid = jnp.concatenate((jnp.ones((indices.size, 1), dtype=jnp.bool_), ~past_terminal), axis=1)
    sequence = jnp.where(valid[..., None], values, 0.)[:, ::-1]
    following = jnp.concatenate((sequence[:, 1:], batch.next_features[:, None]), axis=1)
    return batch._replace(features=sequence, next_features=following)


@lru_cache(maxsize=16)
def make_dsac_update(agent, rollout=ROLLOUT, warmup=0):
    @eqx.filter_jit
    def update(learner, replay, cursor, size, transition, rng):
        transition = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), transition)
        indices = (cursor+jnp.arange(ENVS*rollout)) % CAPACITY
        replay = jax.tree.map(lambda a, b: a.at[indices].set(b), replay, transition)
        size = jnp.minimum(size+ENVS*rollout, CAPACITY)
        cursor = (cursor+ENVS*rollout) % CAPACITY

        def one(learner, key):
            sample_key, update_key = jax.random.split(key)
            if isinstance(agent, RecurrentDSAC):
                available = jnp.maximum(size-ENVS*(models.HISTORY-1), 1)
                age = jax.random.randint(sample_key, (agent.config.minibatch_size,), 0, available)
                idx = (cursor-1-age) % CAPACITY
            else:
                idx = jax.random.randint(sample_key, (agent.config.minibatch_size,), 0, size)
            batch = jax.tree.map(lambda x: x[idx], replay)
            if isinstance(agent, RecurrentDSAC):
                batch = with_history(replay, batch, idx)
            return agent.update(learner, update_key, batch)
        def learn(learner):
            learner, losses = jax.lax.scan(one, learner, jax.random.split(rng, agent.config.gradient_steps))
            return learner, losses.mean(axis=0)
        required = max(warmup,ENVS*models.HISTORY if isinstance(agent,RecurrentDSAC) else 0)
        if required:
            metric_count=7 if isinstance(agent,RecurrentDSAC) else 6
            learner, loss = jax.lax.cond(size >= required, learn,
                lambda learner:(learner,jnp.zeros(metric_count)), learner)
        else:
            learner, loss = learn(learner)
        return learner, replay, cursor, size, loss
    return update


def checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as file:
        pickle.dump(jax.device_get(payload), file, protocol=5)
    temporary.replace(path)


def train(row, steps=None, pilot=False, resume=False, stop_after=None, output_root=None):
    progressive = (budget_curriculum if row['curriculum']==budget_curriculum.NAME else
        ownership_curriculum if row['curriculum']==ownership_curriculum.NAME else
        resource_curriculum if row['curriculum']==resource_curriculum.NAME else gradual)
    design = protocol.load()
    kind = ("dsac_gru" if row["algorithm"] == "hybrid_dsac" and row["architecture"] == "gru"
            else "dsac_mlp" if row["algorithm"] == "hybrid_dsac"
            else "ppo_gru" if row["architecture"] == "gru" else "ppo_mlp")
    final = replace(protocol.run_config(row), loan_capacity=256)
    if row['curriculum'] in (ownership_curriculum.NAME,budget_curriculum.NAME):
        # Allocate enough individual loan records for even a loan every service
        # duration. Allocation is not an economic borrowing limit.
        window=final.horizon or 30*final.day
        final=replace(final,loan_capacity=max(256,(window+final.loan_duration-1)//final.loan_duration+2))
    budget = steps or int(row["physical_transition_budget"])
    decision_budget=int(row.get('network_decision_budget') or 0)
    switch_fraction=float(row.get('dsac_reverse_after_fraction') or 0.)
    switch_stage=int(row.get('dsac_reverse_at_stage') or -1)
    late_reserve=int(row.get('late_reserve_decisions') or 0)
    if switch_fraction and not (isinstance(row.get('dsac_settings'),str) and decision_budget and 0.<switch_fraction<1.):
        raise ValueError('DSAC actor switch requires a decision budget, settings and a fraction in (0, 1)')
    lesson_decision_cap=int(row.get('lesson_decision_cap') or getattr(progressive,'MAX_DECISIONS',0))
    deterministic_readiness=row.get('readiness_inference')=='argmax_mean'
    if decision_budget and decision_budget%(ENVS*ROLLOUT):
        raise ValueError('Network-decision budget must contain complete collection chunks')
    if budget % (ENVS*ROLLOUT):
        raise ValueError(f"Training steps must be divisible by {ENVS*ROLLOUT}")
    validation_stride = max(4096, budget//16)
    output = protocol.ROOT/(output_root or ("runs/pilots" if pilot else "artifacts/training"))/row["run_id"]
    if pilot:
        import hashlib
        identity = {k: v for k, v in row.items() if k != "status"}
        digest = hashlib.sha256(json.dumps({"sources": protocol.source_hashes(), "run": identity,
            "budget": budget, "design": design}, sort_keys=True).encode()).hexdigest()[:10]
        output = output/f"{digest}_{budget}"
    output.mkdir(parents=True, exist_ok=True)
    final_params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), final)
    evaluation_config = replace(final, market="dynamic") if row["training_market"] == "fixed" else final
    evaluation_params = replace(final_params, config=evaluation_config)
    feature_count = strategy.reset(11, final_params)[0].features.size
    rng = jax.random.PRNGKey(int(row["initialization_seed"]))
    rng, init = jax.random.split(rng)
    opt = OPT
    is_dsac = kind.startswith("dsac")
    agent = None
    if is_dsac:
        cfg = dsac.AgentConfig(gamma=final.gamma, demo_weight=0., action_count=strategy.COUNT)
        if row.get('dsac_settings'):
            cfg=replace(cfg,**json.loads(row['dsac_settings']))
        if (switch_fraction or switch_stage>=0) and cfg.categorical_actor_update!='forward_kl':
            raise ValueError('DSAC actor switch must start with forward KL')
        agent = dsac_agent(kind, cfg)
        learner = agent.init(init, feature_count)
        actor = learner.actor
        opt_state = None
    else:
        actor = models.init_ppo(init, feature_count, kind == "ppo_gru", strategy.COUNT)
        learner, opt_state = None, opt.init(actor)
    counters = jnp.arange(ENVS)
    hidden = models.initial_memory(kind, ENVS, feature_count)
    total, index, best, history, stage, replay = 0, 0, None, [], -1, None
    progressive_index, lesson_start, last_gate, final_gate_passed = 0, 0, 0, False
    lesson_start_decisions=0
    is_progressive = row['curriculum'] == progressive.NAME
    restored_stage = -1
    cursor, size = jnp.int32(0), jnp.int32(0)
    intrinsic = None
    intrinsic_opt = None
    if row["exploration"] == "rnd":
        a, b = jax.random.split(jax.random.fold_in(init, 55))
        intrinsic = {"target": dsac._init_mlp(a, (16, 32, 16), .2),
                     "predictor": dsac._init_mlp(b, (16, 32, 16), .2), "weight": jnp.float32(1.)}
        intrinsic_opt = opt.init(intrinsic["predictor"])

    @jax.jit
    def update_intrinsic(intrinsic, state, features, weight):
        def loss(predictor):
            x = jnp.clip(features[..., :16].reshape((-1, 16)), -1., 1.)
            return jnp.mean((dsac._mlp(predictor, x)-dsac._mlp(intrinsic["target"], x))**2)
        grad = jax.grad(loss)(intrinsic["predictor"])
        changes, state = opt.update(grad, state)
        return {**intrinsic, "predictor": optax.apply_updates(intrinsic["predictor"], changes),
                "weight": weight}, state

    algorithm_settings = asdict(agent.config) if is_dsac else {
        "learning_rate": .00025, "gae_lambda": .95, "clip_ratio": final.ppo_clip, "epochs": final.ppo_epochs,
        "target_kl":final.ppo_target_kl,"value_head_lr_multiplier":final.ppo_value_head_lr_multiplier,
        "environment_minibatch": 8, "continuous_entropy": .003, "discrete_entropy": .01,
        "physical_gamma":final.gamma,"decision_discount":"physical_gamma ** actual_duration"}
    metadata = {"kind": kind, "interface": "strategy", "maximum_physical_steps_per_decision": strategy.STEPS,
                "decision_duration":"variable; actual active physical ticks", "macro_names": strategy.NAMES, "environment": asdict(final), "evaluation_environment": asdict(evaluation_config),
                "algorithm_settings": algorithm_settings,
                "actor_parameters": sum(x.size for x in jax.tree.leaves(actor)),
                "validation_interval_physical": validation_stride,
                "run": row, "budget": budget,
                "pilot": pilot, "design": design, "source_hashes": protocol.source_hashes()}
    if resume and (output/"resume.pkl.gz").exists():
        with gzip.open(output/"resume.pkl.gz", "rb") as file:
            saved = pickle.load(file)
        if saved["metadata"]["budget"] != budget:
            raise ValueError("Resume budget differs from the saved run")
        if saved["metadata"]["source_hashes"] != metadata["source_hashes"]:
            raise ValueError("Source files changed since checkpoint; create a new registered run")
        for name in ("environment", "design"):
            if saved["metadata"][name] != metadata[name]:
                raise ValueError(f"Resume {name} differs from the saved experiment")
        old_identity = {k: v for k, v in saved["metadata"]["run"].items() if k != "status"}
        new_identity = {k: v for k, v in row.items() if k != "status"}
        if old_identity != new_identity:
            raise ValueError("Resume initialization or registered treatment changed")
        actor, learner, opt_state, states, hidden, counters, rng, replay = (
            saved[name] for name in ("actor", "learner", "opt_state", "states", "hidden", "counters", "rng", "replay"))
        total, index, best, history, cursor, size, stage = (
            saved[name] for name in ("total", "index", "best", "history", "cursor", "size", "stage"))
        if is_progressive:
            progressive_index, lesson_start, last_gate, final_gate_passed = (
                saved[name] for name in ('progressive_index', 'lesson_start', 'last_gate', 'final_gate_passed'))
            lesson_start_decisions=saved.get('lesson_start_decisions',0)
        intrinsic, intrinsic_opt = saved["intrinsic"], saved["intrinsic_opt"]
        log = output/"training.jsonl"
        if log.exists():
            records = [line for line in log.read_text(encoding="utf-8").splitlines()
                       if json.loads(line)["physical_transitions"] <= total]
            log.write_text(chr(10).join(records)+chr(10), encoding="utf-8")
        if is_progressive:
            gate_log = output/'curriculum_gates.jsonl'
            if gate_log.exists():
                records = [line for line in gate_log.read_text(encoding='utf-8').splitlines()
                    if json.loads(line)['physical_transitions'] <= total]
                gate_log.write_text('\n'.join(records)+('\n' if records else ''), encoding='utf-8')
        restored_stage = stage
        stage = -1  # Preserve restored states only within the same curriculum stage.
        resumed_states = True
    else:
        if (output/"config.json").exists():
            raise FileExistsError("Run already exists; resume it or use another registered run")
        (output/"config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for name in metadata["source_hashes"]:
            destination = output/"source"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(protocol.ROOT/name, destination)
        shutil.copy2(protocol.ROOT/"requirements.txt", output/"source/requirements.txt")
        for name in ("protocol.json", "registry.csv"):
            destination = output/"source/experiments"/name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(protocol.ROOT/"experiments"/name, destination)
        models.save_policy(output/"initial.npz", actor, {**metadata, "physical_transitions": 0})
        states, resumed_states = None, False
    if not pilot and output_root is None:
        protocol.set_run_status(row["run_id"], "running")
    start = perf_counter()
    print(f"START {row['run_id']} {jax.devices()} budget={budget}", flush=True)
    while total < budget and (not decision_budget or index*ENVS*ROLLOUT<decision_budget):
        fraction = total/budget
        current_stage = (progressive_index if is_progressive else
                        sum(fraction>=b for b in (.1,.25,.5,.75)) if row['curriculum']=='cold_debt500' else
                        min(int(fraction*4),3) if row["curriculum"] == "generalist" else
                        7 if row["curriculum"] == "none" or fraction >= .5 else
                         4+min(int((fraction-.2)/.1), 2) if fraction >= .2 else min(int(fraction/.05), 3))
        if stage != current_stage:
            previous_stage = restored_stage if resumed_states else stage
            actual_change = previous_stage >= 0 and previous_stage != current_stage
            if row['curriculum']=='cold_debt500' and actual_change and previous_stage==0:
                records=[json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()]
                totals={name:sum(r.get('completed_actions',{}).get(name,0) for r in records if r['stage']==0)
                    for name in ('paid','meals','sleeps','refills','spins','hands','loans','car_payments','micro_payments')}
                coverage={'training_only':True,'completed_actions':totals,'passed':all(totals.values()),
                    'ownership_events':sum(r.get('ownership_events',0) for r in records if r['stage']==0)}
                coverage['car_loan_closures']=sum(r.get('car_loan_closures',0) for r in records if r['stage']==0)
                coverage['microloan_closures']=sum(r.get('microloan_closures',0) for r in records if r['stage']==0)
                coverage['passed'] &= all(coverage[k]>0 for k in ('ownership_events','car_loan_closures','microloan_closures'))
                (output/'cold_start_coverage.json').write_text(json.dumps(coverage,indent=2))
                if not coverage['passed']:
                    raise RuntimeError('Cold-start coverage incomplete: '+str([k for k,v in totals.items() if not v]))
            if actual_change:
                models.save_policy(output/f"stage_{previous_stage}.npz", actor,
                                   {**metadata, "physical_transitions": total})
                resumed_states = False
            config = protocol.curriculum(final, fraction, row["curriculum"] == "none",
                                         row.get("discount_curriculum") != "constant", strategic=True)
            if row["curriculum"] == "generalist":
                level = min(int(fraction*4),3)
                config = replace(final, starting_debt=(50.,300.,700.,1000.)[level],
                    intermediate_starts=level<3, intermediate_home_fraction=(.25,.5,.75,1.)[level],
                    hunger_tick=final.hunger_tick*(.65,.8,1.,1.)[level],
                    market="fixed" if level==0 else "dynamic")
            if row['curriculum']=='cold_debt500':
                level=sum(fraction>=b for b in (.1,.25,.5,.75))
                config=replace(final,starting_debt=(50.,50.,150.,300.,500.)[level],
                    exploration_epsilon=(.8,.2,0.,0.,0.)[level],exploration_starts=level==0,
                    intermediate_starts=0<level<4,intermediate_home_fraction=(0.,.25,.5,.75,1.)[level],
                    hunger_tick=final.hunger_tick*(.65,.65,.8,1.,1.)[level],
                    market='fixed' if level<2 else 'dynamic')
            if is_progressive:
                config = progressive.configuration(final, current_stage)
            if resumed_states:
                config = replace(config, loan_capacity=states.loan_balance.shape[-1])
            params = replace(final_params, config=config)
            if not resumed_states:
                _, states = reset_batch(params, design, counters)
                hidden = models.initial_memory(kind, ENVS, feature_count)
            resumed_states = False
            collector = compiled_collector(config, kind, design["training_scenario_count"])
            if is_dsac:
                agent = dsac_agent(kind, replace(cfg, gamma=config.gamma))
            update = make_dsac_update(agent,warmup=int(row.get('replay_warmup') or 0)) if is_dsac else compiled_ppo(config,diagnostics=True)
            # Keep off-policy experience across behavior-only exploration changes.
            # Other treatment changes retain the conservative historical reset rule.
            if actual_change and not (is_progressive and progressive.retain_replay(previous_stage, current_stage)):
                replay, cursor, size = None, jnp.int32(0), jnp.int32(0)
            stage = current_stage
        if is_dsac and (switch_fraction or switch_stage>=0):
            reverse=(switch_stage>=0 and stage>=switch_stage) or (switch_fraction and index*ENVS*ROLLOUT>=switch_fraction*decision_budget)
            method='reverse_kl' if reverse else 'forward_kl'
            active_config=replace(cfg,gamma=config.gamma,categorical_actor_update=method)
            if agent.config!=active_config:
                agent=dsac_agent(kind,active_config)
                update=make_dsac_update(agent,warmup=int(row.get('replay_warmup') or 0))
        rng, collect_key, update_key = jax.random.split(rng, 3)
        if intrinsic is not None:
            intrinsic = {**intrinsic, "weight": jnp.float32(max(0, 1.-fraction/.75))}
        if int(np.asarray((states.loan_balance > 0).sum(axis=-1)).max()) > params.config.loan_capacity-ROLLOUT:
            capacity = params.config.loan_capacity*2
            padding = capacity-params.config.loan_capacity
            states = replace(states, **{name: jnp.pad(getattr(states, name), ((0, 0), (0, padding)))
                for name in ("loan_ids", "loan_principal", "loan_balance", "loan_rates", "loan_opened")})
            params = replace(params, config=replace(params.config, loan_capacity=capacity))
            collector = compiled_collector(params.config, kind, design["training_scenario_count"])
        (states, hidden, counters, _), trace = collector(actor, states, hidden, counters, collect_key, intrinsic)
        transition = trace[0]
        if bool(np.any(np.asarray(trace[5]["capacity_needed"]))):
            raise RuntimeError("Loan capacity needs expansion before continuing this collector")
        if is_dsac:
            if replay is None:
                replay = jax.tree.map(lambda x: jnp.zeros((CAPACITY, *x.shape[2:]), x.dtype), transition)
            learner, replay, cursor, size, loss = update(learner, replay, cursor, size, transition, update_key)
            actor = learner.actor
        else:
            actor, opt_state, loss = update(actor, opt_state, trace, states, hidden, update_key)
        if not bool(jnp.all(jnp.isfinite(loss))):
            raise FloatingPointError("Nonfinite training loss")
        if intrinsic is not None:
            features = transition.features
            intrinsic, intrinsic_opt = update_intrinsic(intrinsic, intrinsic_opt, features,
                                                       jnp.float32(max(0, 1.-fraction/.75)))
        physical = flatten_physical(trace[5])
        active_count = int(np.asarray(physical["active"]).sum())
        total += active_count
        log_bytes = write_chunk(output/"traces"/f"chunk_{index:05}.npz", physical,
                                 {"stage": stage, "stage_configuration": asdict(params.config),
                                  "physical_transition_start": total-active_count, "scheduled_physical_slots": ENVS*ROLLOUT*strategy.STEPS})
        index += 1
        loss_value=float(loss[0])
        record = {"physical_transitions": total, "stage": stage, "loss": loss_value,
                  "loss_type": "quantile_critic" if is_dsac else "ppo_objective",
                  "elapsed_seconds": perf_counter()-start, "log_bytes": log_bytes,
                  "scenario_min": int(np.asarray(trace[5]["scenario_id"]).min()),
                  "scenario_max": int(np.asarray(trace[5]["scenario_id"]).max())}
        record['network_decisions_total']=index*ENVS*ROLLOUT
        record['network_decisions_this_chunk']=ENVS*ROLLOUT
        required=max(int(row.get('replay_warmup') or 0),ENVS*models.HISTORY if kind=='dsac_gru' else 0)
        record['optimizer_updates_this_chunk']=(agent.config.gradient_steps if int(size)>=required else 0) if is_dsac else int(loss[1])
        if not is_dsac:
            record['maximum_minibatch_kl']=float(loss[2])
            record['explained_variance']=float(loss[3])
        if row['curriculum']=='cold_debt500' or is_progressive:
            active=np.asarray(physical['active'])
            counts=np.asarray(physical['events'])[active].sum(axis=0)
            record['completed_actions']=dict(zip(env.COUNTER_NAMES,map(int,counts)))
            record['ownership_events']=int(((np.asarray(physical['outcome'])==env.SUCCESS)&active).sum())
            record['car_loan_closures']=int((active&(np.asarray(physical['car_payment'])>0)&
                (np.asarray(physical['car_cleared_at'])==np.asarray(physical['time']))).sum())
            record['microloan_closures']=int((active&(np.asarray(physical['micro_payment'])>0)&
                (np.asarray(physical['micro_debt'])==0)).sum())
            record['exploration_epsilon']=params.config.exploration_epsilon
        if is_dsac:
            record['categorical_actor_update']=agent.config.categorical_actor_update
            names = ("critic_loss", "actor_loss", "continuous_temperature", "discrete_temperature",
                     "continuous_sample_entropy", "discrete_entropy", "encoder_loss")
            record.update({name: float(value) for name, value in zip(names, np.asarray(loss))})
        with (output/"training.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record)+chr(10))
        curriculum_failure = None
        gate_ran = False
        qualified_selection = getattr(progressive,'SELECT_QUALIFIED_CHECKPOINT',False) and is_progressive
        budget_ending = total>=budget or (decision_budget and index*ENVS*ROLLOUT>=decision_budget)
        if (is_progressive and not final_gate_passed and total-lesson_start >= progressive.MIN_PHYSICAL
                and index*ENVS*ROLLOUT-lesson_start_decisions >= getattr(progressive,'MIN_DECISIONS',0)
                and (total-last_gate >= progressive.GATE_INTERVAL or budget_ending or
                     (late_reserve and stage<13 and index*ENVS*ROLLOUT>=decision_budget-late_reserve))):
            gate_ran = True
            last_gate = total
            exposure = (cold_coverage(output)['passed'] if stage == 0 and
                getattr(progressive,'REQUIRE_COLD_COVERAGE',True) else True)
            gate_params = replace(params, config=replace(params.config, exploration_epsilon=0.))
            if hasattr(progressive,'evaluation_configuration'):
                gate_params=replace(gate_params,config=progressive.evaluation_configuration(gate_params.config))
            gate_summary, gate_states, _ = evaluate(actor, kind, gate_params, progressive.VALIDATION_IDS,
                output/'curriculum_validation'/f'stage_{stage:02}_step_{total:09}',
                deterministic=deterministic_readiness,interface='strategy')
            paid_index = env.COUNTER_NAMES.index('paid')
            earned_successes = int(((np.asarray(gate_states.outcome) == env.SUCCESS) &
                (np.asarray(gate_states.counters)[:, paid_index] > 0)).sum())
            extra={}
            if getattr(progressive,'EXTRA_GATE_METRICS',False):
                success=np.asarray(gate_states.outcome)==env.SUCCESS
                counts=np.asarray(gate_states.counters)
                extra['successful_meal_episodes']=int((success&(counts[:,env.MEALS]>0)).sum())
                extra['successful_refuel_episodes']=int((success&(counts[:,env.REFILLS]>0)).sum())
            decision = progressive.gate(stage, gate_summary, earned_successes, exposure,**extra)
            if decision['passed'] and hasattr(progressive,'additional_checks'):
                retained=[]
                for name,context,check_config,check_ids in progressive.additional_checks(final,stage):
                    check_params=replace(final_params,config=check_config)
                    check_summary,check_states,_=evaluate(actor,kind,check_params,check_ids,
                        output/'curriculum_validation'/f'stage_{stage:02}_step_{total:09}'/name,
                        deterministic=deterministic_readiness,interface='strategy')
                    success=np.asarray(check_states.outcome)==env.SUCCESS
                    check_counts=np.asarray(check_states.counters)
                    result=progressive.gate(context,check_summary,
                        int((success&(check_counts[:,env.PAID]>0)).sum()),True,
                        successful_meal_episodes=int((success&(check_counts[:,env.MEALS]>0)).sum()),
                        successful_refuel_episodes=int((success&(check_counts[:,env.REFILLS]>0)).sum()))
                    retained.append({'name':name,'context':context,'summary':check_summary,**result})
                decision['additional_checks']=retained
                if not all(x['passed'] for x in retained):
                    decision['passed']=False
                    decision['failures'].append('current_second_bank_or_retained_skill_failed')
            if qualified_selection and decision['passed'] and stage==len(progressive.LESSONS)-1:
                standard,_,_=evaluate(actor,kind,evaluation_params,range(1000001,1000065),
                    output/'curriculum_validation'/f'stage_{stage:02}_step_{total:09}'/'standard_64',
                    deterministic=deterministic_readiness,interface='strategy')
                decision['standard_validation']=standard
                if standard['ownership']<progressive.STANDARD_OWNERSHIP_MIN:
                    decision['passed']=False
                    decision['failures'].append('standard_ownership_below_threshold')
                else:
                    models.save_policy(output/'selected.npz',actor,
                        {**metadata,'physical_transitions':total,'validation':standard,'curriculum_gate':decision})
            decision.update(stage=stage, lesson=progressive.LESSONS[stage].name,
                physical_transitions=total, stage_physical_transitions=total-lesson_start,
                summary=gate_summary)
            with (output/'curriculum_gates.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(decision)+'\n')
            print('CURRICULUM_GATE '+json.dumps(decision), flush=True)
            if decision['passed']:
                models.save_policy(output/f'lesson_{stage:02}_passed.npz', actor,
                    {**metadata, 'physical_transitions': total, 'curriculum_gate': decision})
                if stage < len(progressive.LESSONS)-1:
                    progressive_index += 1
                    lesson_start = total
                    lesson_start_decisions=index*ENVS*ROLLOUT
                else:
                    final_gate_passed = True
            elif (total-lesson_start >= progressive.MAX_PHYSICAL and
                  index*ENVS*ROLLOUT-lesson_start_decisions>=lesson_decision_cap):
                curriculum_failure = 'Actor failed its lesson gate after both physical and decision exposure caps'
        decision_stride=max(ENVS*ROLLOUT,decision_budget//16)
        if late_reserve and progressive_index<13 and index*ENVS*ROLLOUT>=decision_budget-late_reserve:
            curriculum_failure='Early curriculum did not qualify before the reserved late-stage budget'
        validation_due = (index == 1 or not history or total>=budget or curriculum_failure is not None or
            (index*ENVS*ROLLOUT >= (int(history[-1].get('network_decisions',0))//decision_stride+1)*decision_stride
                if decision_budget and history else
             bool(history) and total >= (int(history[-1]['physical_transitions'])//validation_stride+1)*validation_stride) or
            (decision_budget and index*ENVS*ROLLOUT>=decision_budget) or (stop_after and total>=stop_after))
        if gate_ran or validation_due:
            if validation_due:
                validation, _, _ = evaluate(actor, kind, evaluation_params, range(1000001, 1000065),
                                            output/"validation_traces"/f"step_{total:09}",
                                            deterministic=deterministic_readiness,interface="strategy")
                history.append({"physical_transitions": total,"network_decisions":index*ENVS*ROLLOUT, **validation})
                score = (validation["mean_external_return"], validation["success_rate"], -(validation["starvation"]+validation["collision"]),
                         -validation["mean_remaining_debt"], -validation["mean_endpoint_adjusted_completion_tick"])
                if best is None or score > best:
                    best = score
                    selection_name="best_unqualified" if qualified_selection else "selected"
                    models.save_policy(output/(selection_name+".npz"), actor,
                                       {**metadata, "physical_transitions": total, "validation": validation})
                    checkpoint(output/(selection_name+"_state.pkl.gz"), {"metadata": metadata, "total": total,
                        "actor": actor, "learner": learner, "training_configuration": asdict(params.config)})
            models.save_policy(output/"last.npz", actor, {**metadata, "physical_transitions": total})
            checkpoint(output/f"learner_{total:09}.pkl.gz", {"metadata": metadata, "total": total,
                "actor": actor, "learner": learner, "opt_state": opt_state,
                "training_configuration": asdict(params.config)})
            (output/"validation.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint(output/"resume.pkl.gz", {
                "metadata": metadata, "actor": actor, "learner": learner, "opt_state": opt_state,
                "states": states, "hidden": hidden, "counters": counters, "rng": rng, "replay": replay,
                "total": total, "index": index, "best": best, "history": history, "cursor": cursor,
                "size": size, "stage": stage, "intrinsic": intrinsic, "intrinsic_opt": intrinsic_opt,
                "progressive_index": progressive_index, "lesson_start": lesson_start,
                "last_gate": last_gate, "final_gate_passed": final_gate_passed,
                "lesson_start_decisions":lesson_start_decisions})
            if validation_due:
                print(f"{row['run_id']} {total}/{budget} validation={validation['ownership']}/64 "
                      f"death={validation['starvation']} crash={validation['collision']} debt={validation['mean_remaining_debt']:.1f}",
                      flush=True)
                if decision_budget:
                    print(f"NETWORK_DECISIONS {row['run_id']} {index*ENVS*ROLLOUT}/{decision_budget}",flush=True)
        if curriculum_failure:
            (output/'curriculum_failure.json').write_text(json.dumps({
                'reason': curriculum_failure, 'stage': stage, 'physical_transitions': total,
                'network_decisions':index*ENVS*ROLLOUT}, indent=2))
            raise RuntimeError(curriculum_failure)
        if stop_after and total >= stop_after and total < budget:
            return output
    if is_progressive and not final_gate_passed:
        reason = {'reason': 'Total budget exhausted before all actor-only curriculum gates passed',
            'stage': stage, 'physical_transitions': total,'network_decisions':index*ENVS*ROLLOUT}
        (output/'curriculum_failure.json').write_text(json.dumps(reason, indent=2))
        raise RuntimeError(reason['reason'])
    if not pilot and output_root is None:
        protocol.set_run_status(row["run_id"], "trained")
    models.save_policy(output/"final.npz", actor, {**metadata, "physical_transitions": total})
    (output/"complete.json").write_text(json.dumps({"physical_transitions": total,
        "network_decisions":index*ENVS*ROLLOUT,
        "elapsed_seconds": perf_counter()-start, "best_validation": best}), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--group", choices=("main", "extension", "all"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Save a resumable checkpoint at this transition count")
    parser.add_argument("--init-seed", type=int, help="Override initialization for an explicitly labeled pilot only")
    args = parser.parse_args()
    if args.init_seed is not None and not args.pilot:
        parser.error("Initialization overrides are allowed only for --pilot")
    rows = protocol.registry()
    chosen = [r for r in rows if r["run_id"] == args.run_id] if args.run_id else [
        r for r in rows if args.group == "all" or r["group"] == args.group]
    if not chosen:
        parser.error("Select an existing --run-id or a --group")
    for row in chosen:
        if args.init_seed is not None:
            row = {**row, "initialization_seed": str(args.init_seed)}
        target = protocol.ROOT/"artifacts/training"/row["run_id"]/"complete.json"
        if target.exists() and not args.pilot:
            continue
        train(row, args.steps, args.pilot, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
