from pathlib import Path
import hashlib,json
root=Path('runs/pilots/dsac_mlp_10d_curriculum_dynamic_repA/350718eb27_2097152')
meta=json.loads((root/'config.json').read_text(encoding='utf-8'))
recovered=[]; failed=[]
old_potential='''    useful = jnp.array((state.fatigue > .3, work_open, True, state.car_debt+micro_debt(state) == 0,
                        True, True, state.fuel < 8., state.satiety < .5,
                        state.satiety < .5, state.satiety < .5))
    # Once settlement is affordable, supply dense progress toward the task's
    # financial and physical endpoint. The policy still chooses every action.
    debt = state.car_debt+micro_debt(state)
    settlement = jnp.zeros(10, dtype=jnp.bool_).at[
        jnp.where(debt == 0, city.GARAGE, city.BANK)].set(True)
    useful = jnp.where((debt == 0) | (state.cash >= debt+20.), settlement, useful)
    nearby = jnp.min(jnp.where(useful, distance, 80.))
    value = .4*progress+.035*state.satiety+.02*state.fuel/params.config.tank-.02*state.fatigue-.05*nearby/40.
'''
for name,expected in meta['source_hashes'].items():
 existing=root/'source'/name
 if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest()==expected:
  recovered.append(name); continue
 raw=Path(name).read_bytes()
 s=raw.decode('utf-8').replace('\r\n','\n')
 if name.endswith('hybrid_dsac.py'):
  s=s.replace('    continuous_temperature: float = .000006\n    discrete_temperature: float = .000012\n','').replace('jnp.array((c.continuous_temperature, c.discrete_temperature))','jnp.array((0.006, 0.012))').replace('updates), -16., -10.)','updates), -9., -2.)')
 if name.endswith('resource_taxi.py'):
  a=s.index('    debt = state.car_debt+micro_debt(state)',s.index('def potential(')); b=s.index('    return jnp.where(state.outcome',a); s=s[:a]+old_potential+s[b:]
 if name=='src/models.py':
  a=s.index('    if key.ndim == 2:'); b=s.index('    return env.Action',a); s=s[:a]+'    control, choice, raw, lp, hc, hd = sample_hybrid(key, mean, std, logp, deterministic)\n'+s[b:]
 if name=='src/evaluate.py':
  a=s.index('            key = jax.vmap('); b=s.index('            if kind ==',a); s=s[:a]+'            key = jax.random.fold_in(jax.random.PRNGKey(9181), start+index)\n'+s[b:]
  s=s.replace('                key, states, actions, params)','                jax.random.split(key, count), states, actions, params)').replace('output=None, deterministic=False,','output=None, deterministic=True,')
  s='\n'.join(line for line in s.split('\n') if 'parser.add_argument("--deterministic"' not in line and 'deterministic=args.deterministic,' not in line)
 if name=='src/train.py':
  s=s.replace('import shutil\n','').replace('score = (validation["mean_external_return"], validation["success_rate"],','score = (validation["success_rate"],')
  a=s.index('        for name in metadata["source_hashes"]:'); b=s.index('        models.save_policy(output/"initial.npz"',a); s=s[:a]+s[b:]
 if name=='src/run_suite.py':
  s=s.replace('    yield "deterministic", config, "test_standard", "learned"\n','').replace('list(variants(config))[:2]','list(variants(config))[:1]').replace('controller=controller, deterministic=label == "deterministic"','controller=controller')
 if name=='src/policy_probes.py':
  s=s.replace('env.Config(**metadata.get("evaluation_environment", metadata["environment"]))','env.Config(**metadata["environment"])')
 if name=='src/build_reports.py':
  s=s.replace('start at 0.000006 and 0.000012','start at 0.006 and 0.012').replace('are clipped to [-16,-10]','are clipped to [-9,-2]').replace('mean discounted external return, ownership, survival, remaining debt and completion time','ownership, survival, remaining debt and completion time').replace('Sampling the trained stochastic policy is primary, using a separate policy RNG keyed by scenario and absolute tick. Argmax/mean extraction is a separately labeled diagnostic.','Deterministic policy evaluation is primary.').replace('Dense feedback has no casino-directed target or visit bonus and is potential-based:','Dense feedback is potential-based:')
  s=s.replace('An initial full-budget engineering pilot failed to learn: all deterministic validation runs ended in starvation or collision and no work was completed. A sampled-policy diagnostic also failed. Calibration reduced DSAC temperatures, increased bounded navigation shaping without changing the external objective, removed casino destinations from shaping, and aligned checkpoint selection with external return. These changes were evaluated together. ','')
 choices=[raw,s.encode('utf-8'),s.replace('\n','\r\n').encode('utf-8')]
 match=next((x for x in choices if hashlib.sha256(x).hexdigest()==expected),None)
 if match is None: failed.append(name); continue
 target=root/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(match); recovered.append(name)
(root/'source_recovery.json').write_text(json.dumps({'exact_hash_matches':recovered,'unrecovered':failed},indent=2),encoding='utf-8')
print('Recovered',len(recovered),'unrecovered',failed)
