"""Blackjack with replacement, soft aces, and an explicit player decision."""

import jax
import jax.numpy as jnp


MAX_CARDS = 22


def total(hand):
    hard = jnp.minimum(hand, 10).sum(axis=-1)
    soft = jnp.any(hand == 1, axis=-1) & (hard <= 11)
    return hard+10*soft, soft


def resolve(key, hand, dealer, dealing, hitting, standing, stake, commission=.10):
    ranks = jax.random.randint(key, (2, MAX_CARDS), 1, 14)
    hand = jnp.where(dealing, jnp.zeros_like(hand).at[:2].set(ranks[0, :2]), hand)
    dealer = jnp.where(dealing, jnp.zeros_like(dealer).at[:2].set(ranks[1, :2]), dealer)
    index = jnp.minimum(jnp.count_nonzero(hand), MAX_CARDS-1)
    hand = hand.at[index].set(jnp.where(hitting, ranks[0, index], hand[index]))
    player, _ = total(hand)
    natural = dealing & (player == 21)
    dealer_natural = dealing & (total(dealer)[0] == 21)
    finished = standing | (hitting & (player >= 21)) | natural | dealer_natural

    def draw(i, current):
        return current.at[i].set(jnp.where(total(current)[0] < 17, ranks[1, i], 0))

    final_dealer = jax.lax.fori_loop(2, MAX_CARDS, draw, dealer)
    dealer_total, _ = total(final_dealer)
    won = (player <= 21) & ((dealer_total > 21) | (player > dealer_total) |
                            (natural & ~dealer_natural)) & ~dealer_natural
    push = (player <= 21) & ((player == dealer_total) | (natural & dealer_natural))
    push &= ~dealer_natural | natural
    profit = jnp.where(natural, 1.5, 1.)*(1.-commission)
    payout = stake*jnp.where(won, 1.+profit, jnp.where(push, 1., 0.))
    return hand, dealer, finished, jnp.where(finished, payout, 0.)
