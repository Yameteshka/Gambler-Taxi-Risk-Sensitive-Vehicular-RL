"""Independent motor-only RL diagnostic; no economic action demonstrations.

This supplementary experiment does not alter the registered economic tasks.
All navigation labels are terminal/reward events, never target control labels.
"""
from dataclasses import replace, asdict
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
import argparse
import gzip
import hashlib
import json
import pickle
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from env import resource_taxi as env, city
import models
import protocol
from algorithms.hybrid_dsac import Transition
from telemetry import frame, write_chunk

ENVS, CHUNK, GAMMA = 64, 64, .99
TRAIN_COUNT = protocol.load()["training_scenario_count"]
CONFIG = env.Config(days=0, installments=False, hunger_tick=0., hunger_work=0., hunger_drive=0.,
                    hunger_push=0., fuel_distance=0., fuel_work=0., car_daily_rate=0.,
                    micro_daily_rate=0., financial_progress=False, gamma=GAMMA)
OPT = optax.chain(optax.clip_by_global_norm(.5), optax.adam(2.5e-4))


class MotorState(NamedTuple):
    game: env.State
    target: jax.Array
    elapsed: jax.Array


def reset(scenario, params, level):
    _, game = env.ResourceTaxiEnv.reset(scenario, params)
    u = jax.random.uniform(env.event_key(game, 61, 0), (10,))
    target = jax.random.randint(env.event_key(game, 60, 0), (), 0, 10)
    maximum = jnp.array((2.1, 4., 8., 16., 28.))[level]
    radius = 1.56+(maximum-1.56)*u[0]**2
    angles = jax.random.uniform(env.event_key(game, 62, 0), (64,), minval=-jnp.pi, maxval=jnp.pi)
    positions = params.zone_centers[target]-radius*jnp.stack((jnp.cos(angles), jnp.sin(angles)), -1)
    obstacles = jnp.concatenate((game.static_obstacles, game.kinematic_obstacles))
    valid = jnp.linalg.norm(positions[:, None]-obstacles[None], axis=-1).min(axis=1) > 2.2
    valid &= jnp.all((positions > 1.2) & (positions < jnp.array((city.WIDTH-2.2, city.HEIGHT-2.2))), axis=1)
    index = jnp.argmax(valid)
    position = jnp.where(jnp.any(valid), positions[index], params.zone_centers[target]-jnp.array((1.6, 0.)))
    direction = params.zone_centers[target]-position
    heading = jnp.arctan2(direction[1], direction[0])+(2*u[1]-1)*jnp.array((.35, .6, 1., 1.5, jnp.pi))[level]
    game = replace(game, agent_pos=position, agent_forward_dir=jnp.array((jnp.cos(heading), jnp.sin(heading))),
        cash=20.+980*u[2], car_debt=1500*u[3], fatigue=u[4], satiety=.4+.6*u[5], fuel=5.+20*u[6],
        loan_balance=game.loan_balance.at[0].set(400*u[7]),
        loan_ids=game.loan_ids.at[0].set(1), loan_principal=game.loan_principal.at[0].set(400*u[7]),
        time=jnp.int32(154*u[8]), shift_claimed=u[9] > .5)
    return MotorState(game, target, jnp.int32(0))


def observation(state, params):
    obs = env.ResourceTaxiEnv.get_observation(state.game, params)
    bearings = obs.features[46:66].reshape((10, 2))
    cue = jnp.concatenate((bearings[state.target], obs.features[36+state.target][None]))
    return replace(obs, features=obs.features.at[-3:].set(cue))


def forced_logp(logp):
    return jnp.full_like(logp, -1e9).at[..., env.DRIVE].set(0.)


@lru_cache(maxsize=5)
def functions(level):
    params, _ = env.ResourceTaxiEnv.init_params(jax.random.key(0), CONFIG)
    reset_batch = jax.vmap(reset, in_axes=(0, None, None))
    observe = jax.vmap(observation, in_axes=(0, None))

    @eqx.filter_jit
    def collector(model, states, counters, rng):
        def tick(carry, _):
            states, counters, rng = carry
            rng, key = jax.random.split(rng)
            obs = observe(states, params)
            mean, std, logp, value, _ = models.ppo_step(model, obs.features, obs.action_mask, jnp.zeros((ENVS, 128)))
            control, choice, raw, lp, _, _ = models.sample_hybrid(key, mean, std, forced_logp(logp))
            action = env.Action(control, choice)
            _, new, _, _, info = jax.vmap(env.ResourceTaxiEnv.step, in_axes=(0, 0, 0, None))(
                jax.random.split(key, ENVS), states.game, action, params)
            elapsed = states.elapsed+1
            arrived = (jnp.linalg.norm(new.agent_pos-params.zone_centers[states.target], axis=-1) <= CONFIG.zone_radius) & (new.speed < CONFIG.parking_speed)
            crashed = new.outcome == env.CRASH
            timed_out = elapsed >= 192
            done = arrived | crashed | timed_out
            outcome = jnp.where(crashed, env.CRASH, jnp.where(arrived, env.SUCCESS, jnp.where(timed_out, env.TIMEOUT, env.ACTIVE)))
            new = replace(new, outcome=outcome)
            next_states = MotorState(new, states.target, elapsed)
            phi_old = -.04*jax.vmap(env.distances, in_axes=(0, None))(states.game.agent_pos, params)[jnp.arange(ENVS), states.target]
            phi_new = -.04*jax.vmap(env.distances, in_axes=(0, None))(new.agent_pos, params)[jnp.arange(ENVS), states.target]
            phi_new = jnp.where(done, 0., phi_new)
            base = jnp.where(arrived & ~crashed, 1., jnp.where(done, -.1, -.1*(1-GAMMA)))
            reward = base+GAMMA*phi_new-phi_old
            following = observe(next_states, params)
            transition = Transition(obs.features, obs.action_mask, control, choice, reward, done, following.features, following.action_mask)
            telemetry = frame(states.game, new, action, reward, info)
            telemetry["external_reward"] = base
            telemetry["shaping"] = GAMMA*phi_new-phi_old
            telemetry["financial_reward"] = jnp.zeros(ENVS)
            telemetry["motor_target"] = states.target
            telemetry["motor_elapsed"] = elapsed
            telemetry["motor_base_reward"] = base
            counters += ENVS*done
            initial = reset_batch(11+counters%TRAIN_COUNT, params, level)
            next_states = jax.tree.map(lambda a,b:jnp.where(done.reshape((ENVS,)+(1,)*(a.ndim-1)),b,a), next_states, initial)
            return (next_states, counters, rng), (transition, value, raw, lp, telemetry)
        return jax.lax.scan(tick, (states, counters, rng), jnp.arange(CHUNK))

    @eqx.filter_jit
    def update(model, optimizer, trace, states, rng):
        transitions, values, raw, old_logp, _ = trace
        obs = observe(states, params)
        _, _, _, last_value, _ = models.ppo_step(model, obs.features, obs.action_mask, jnp.zeros((ENVS,128)))
        next_values = jnp.concatenate((values[1:], last_value[None]))
        def gae(carry, row):
            reward, done, value, next_value = row
            discount = GAMMA*~done
            carry = reward+discount*next_value-value+discount*.95*carry
            return carry, carry
        _, advantage = jax.lax.scan(gae, jnp.zeros(ENVS), (transitions.reward, transitions.done, values, next_values), reverse=True)
        returns = advantage+values
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-6)
        def minibatch(carry, indices):
            model, optimizer = carry
            noise = jax.random.normal(jax.random.fold_in(rng, indices[0]+731), raw[:,indices].shape)
            def loss(candidate):
                mean, std, logp, prediction, _ = models.ppo_step(candidate, transitions.features[:,indices], transitions.mask[:,indices], jnp.zeros((CHUNK,8,128)))
                logp = forced_logp(logp)
                lp, _, _ = models.hybrid_logprob(mean, std, logp, raw[:,indices], transitions.choice[:,indices])
                ratio = jnp.exp(lp-old_logp[:,indices])
                surrogate = jnp.minimum(ratio*advantage[:,indices], jnp.clip(ratio,.8,1.2)*advantage[:,indices])
                entropy, _ = models.hybrid_entropy(mean, std, logp, noise)
                return -surrogate.mean()+.5*((prediction-returns[:,indices])**2).mean()-.003*entropy.mean()
            value, grad = jax.value_and_grad(loss)(model)
            changes, optimizer = OPT.update(grad, optimizer)
            return (optax.apply_updates(model, changes), optimizer), value
        indices = jax.vmap(lambda k:jax.random.permutation(k,ENVS))(jax.random.split(rng,3)).reshape((-1,8))
        (model, optimizer), losses = jax.lax.scan(minibatch,(model,optimizer),indices)
        return model, optimizer, losses.mean()
    return params, reset_batch, collector, update


def run(steps, seed):
    if steps%4096:
        raise ValueError("Budget must be divisible by 4096")
    sources = {**protocol.source_hashes(), "experiments/motor_pretraining.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "experiments/protocol.json": hashlib.sha256((ROOT/"experiments/protocol.json").read_bytes()).hexdigest(),
               "requirements.txt": hashlib.sha256((ROOT/"requirements.txt").read_bytes()).hexdigest()}
    design = {"task":"motor_only_RL", "steps":steps,"seed":seed,"sources":sources,"config":asdict(CONFIG),
              "algorithm":"PPO conditional on Drive; categorical head receives no supervision", "economic_demonstrations":False,
              "observations":"Public game features, randomized independent finances, and a random-goal route cue", "horizon":192,
              "training_scenarios": [11, 10+TRAIN_COUNT], "role":"Supplementary engineering diagnostic, not a registered full-game outcome."}
    digest = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()[:10]
    folder = ROOT/"runs/motor"/f"{digest}_{steps}"
    folder.mkdir(parents=True,exist_ok=True)
    params, reset_batch, _, _ = functions(0)
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    features = observation(reset(jnp.int32(11),params,0),params).features.size
    model = models.init_ppo(key,features,False)
    optimizer = OPT.init(model)
    counters = jnp.arange(ENVS)
    states = reset_batch(11+counters,params,0)
    total, stage = 0, -1
    if (folder/"resume.pkl.gz").exists():
        with gzip.open(folder/"resume.pkl.gz","rb") as file: saved=pickle.load(file)
        model,optimizer,counters,states,rng,total,stage = (saved[k] for k in ("model","optimizer","counters","states","rng","total","stage"))
        if saved["design"] != design: raise ValueError("Motor source/configuration changed")
        log=folder/"learning.jsonl"
        if log.exists():
            kept=[line for line in log.read_text(encoding="utf-8").splitlines() if json.loads(line)["physical_transitions"]<=total]
            log.write_text("\n".join(kept)+"\n",encoding="utf-8")
    else:
        (folder/"design.json").write_text(json.dumps(design,indent=2),encoding="utf-8")
        for name in sources:
            target=folder/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    print(f"START motor-only PPO {folder.name} budget={steps}",flush=True)
    started=perf_counter()
    while total<steps:
        level=min(int(total/steps*5),4)
        params,reset_batch,collect,update=functions(level)
        if stage!=level:
            states=reset_batch(11+counters%TRAIN_COUNT,params,level);stage=level
        rng, key, optkey=jax.random.split(rng,3)
        (states,counters,_),trace=collect(model,states,counters,key)
        model,optimizer,loss=update(model,optimizer,trace,states,optkey)
        if not np.isfinite(float(loss)):raise FloatingPointError("Nonfinite motor loss")
        total+=4096
        write_chunk(folder/"traces"/f"chunk_{total//4096:05}.npz",trace[-1],{"task":"motor_only","level":level,"physical_transition_end":total})
        record={"physical_transitions":total,"level":level,"loss":float(loss),"ownership_label_means_motor_arrival":int((np.asarray(trace[-1]["outcome"])==env.SUCCESS).sum()),
                "collisions":int((np.asarray(trace[-1]["outcome"])==env.CRASH).sum()),"timeouts":int((np.asarray(trace[-1]["outcome"])==env.TIMEOUT).sum()),"elapsed_seconds":perf_counter()-started}
        with (folder/"learning.jsonl").open("a",encoding="utf-8") as file:file.write(json.dumps(record)+"\n")
        if total%131072==0 or total==steps:
            models.save_policy(folder/"last.npz",model,{"kind":"ppo_mlp","role":"motor_only_RL_not_full_game","physical_transitions":total,"design":design})
            with gzip.open(folder/"resume.tmp","wb",compresslevel=1) as file:pickle.dump(jax.device_get({"model":model,"optimizer":optimizer,"counters":counters,"states":states,"rng":rng,"total":total,"stage":stage,"design":design}),file,protocol=5)
            (folder/"resume.tmp").replace(folder/"resume.pkl.gz")
            print(json.dumps(record),flush=True)
    (folder/"complete.json").write_text(json.dumps({"steps":total,"role":"motor_only_not_economic_result"}),encoding="utf-8")
    return folder


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=1048576)
    parser.add_argument("--seed",type=int,default=101)
    args=parser.parse_args()
    print(run(args.steps,args.seed),flush=True)
