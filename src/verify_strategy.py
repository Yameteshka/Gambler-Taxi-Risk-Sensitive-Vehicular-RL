"""Independent checks of intentions, physical timing, transactions and discount."""
from dataclasses import replace
import json
import equinox as eqx
import jax,jax.numpy as jnp,numpy as np
from env import resource_taxi as env, city
import strategy,protocol


def verify():
 params,_=env.ResourceTaxiEnv.init_params(jax.random.key(0))
 _,initial=strategy.reset(6,params)
 advance=eqx.filter_jit(strategy.advance)
 def command(state,choice):return advance(state,env.Action(jnp.zeros(2),jnp.int32(choice)),params)
 bank=replace(initial,agent_pos=params.zone_centers[city.BANK])
 _,borrowed,reward,_,rows=command(bank,strategy.BORROW_25)
 assert borrowed.counters[env.LOANS]==1
 assert int(np.asarray(rows["loan_opened"]>0).sum())==1
 np.testing.assert_allclose(borrowed.cash,125.)
 assert borrowed.time==params.config.loan_duration and env.micro_debt(borrowed)>25
 np.testing.assert_allclose(reward,(np.asarray(rows["reward"])*params.config.gamma**np.arange(strategy.STEPS)).sum(),atol=1e-6)
 telescoped=params.config.gamma**int(borrowed.time)*env.potential(borrowed,params)-env.potential(bank,params)
 np.testing.assert_allclose((np.asarray(rows["shaping"])*params.config.gamma**np.arange(strategy.STEPS)).sum(),telescoped,atol=2e-6)
 work=replace(initial,agent_pos=params.zone_centers[city.WORK],fatigue=jnp.float32(1.))
 _,working,_,_,rows=command(work,strategy.WORK)
 assert (np.asarray(rows["executed"])==env.WORK).sum()==1
 assert working.shift_claimed and working.remaining==0 and working.wages>0
 poor=replace(initial,cash=jnp.float32(0.))
 _,waiting,_,_,_=command(poor,strategy.WAIT)
 assert waiting.counters[env.LOANS]==0 and waiting.cash==0
 slots=replace(initial,agent_pos=params.zone_centers[city.SLOTS],time=jnp.int32(100))
 _,spinning,_,_,_=command(slots,strategy.SLOTS_SMALL)
 assert spinning.counters[env.SPINS]==1
 _,after,_,_,_=command(spinning,strategy.WAIT)
 assert after.counters[env.SPINS]==1 and after.round_count==1
 hand=replace(initial,agent_pos=params.zone_centers[city.BLACKJACK],activity=jnp.int32(env.DEAL_SMALL),
              hand=initial.hand.at[:2].set(jnp.array((10,6))),dealer=initial.dealer.at[:2].set(jnp.array((10,7))))
 assert env.legal(hand,params)[env.WAIT]
 _,thinking,_,_,_=command(hand,strategy.WAIT)
 np.testing.assert_array_equal(thinking.hand,hand.hand)
 np.testing.assert_array_equal(thinking.agent_pos,hand.agent_pos)
 assert thinking.satiety<hand.satiety
 # EAT is a shared physical action; a cafe intention must not purchase at the
 # budget diner merely because that other venue is already within reach.
 diner=replace(initial,agent_pos=params.zone_centers[city.DINER])
 _,toward_cafe,_,_,cafe_rows=command(diner,strategy.CAFE)
 assert not np.any((np.asarray(cafe_rows["executed"])==env.EAT)&(np.asarray(cafe_rows["meal_venue"])==0))
 assert int(np.asarray(rows["active"]).sum())==params.config.shift_duration
 settled=replace(initial,car_debt=jnp.float32(0.))
 assert strategy.legal(settled,params)[strategy.BUY]
 assert not strategy.legal(borrowed,params)[strategy.BUY]
 # Administrative censoring must neither move nor charge inactive tail frames.
 _,censored,_,_,tail=advance(bank,env.Action(jnp.zeros(2),jnp.int32(strategy.BORROW_25)),params,stop_tick=3)
 assert censored.time==3
 assert not np.asarray(tail["active"])[3:].any()
 assert not np.asarray(tail["external_reward"])[3:].any()
 assert not np.asarray(tail["micro_payment"])[3:].any()
 result={"passed":True,"source_hashes":protocol.source_hashes(),"checks":["one transaction per intention","physical-time and semi-MDP discount","potential telescoping","full-fatigue work","no forced borrowing","one voluntary wager","blackjack waiting without movement","ownership eligibility","administrative censoring"]}
 (protocol.ROOT/"experiments/strategy_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 print("Strategy timing, transactions, rewards and eligibility checks passed",flush=True)


if __name__=="__main__":verify()
