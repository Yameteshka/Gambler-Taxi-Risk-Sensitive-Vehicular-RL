"""Quantile DSAC conditioned on a learned finite history encoder."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from algorithms import hybrid_dsac as base
import models


class Learner(NamedTuple):
    inner: base.Learner
    encoder: tuple
    target_encoder: tuple
    encoder_opt: object
    actor_encoder: tuple | None = None
    actor_encoder_opt: object | None = None

    @property
    def actor(self):
        return {"actor":self.inner.actor,"encoder":self.encoder if self.actor_encoder is None else self.actor_encoder}


class RecurrentDSAC:
    def __init__(self, config):
        self.config = config
        self.base = base.HybridDSAC(config)
        self.opt = optax.chain(optax.clip_by_global_norm(5.), optax.adam(config.learning_rate))
        self.actor_encoder_optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adam(config.actor_learning_rate))

    def init(self, key, feature_count):
        a, b = jax.random.split(key)
        encoder = models.init_gru(a, feature_count)
        # Clone the initial weights for a matched shared/separate comparison;
        # immutable updates and independent optimizers untie them immediately.
        actor_encoder=encoder if self.config.separate_recurrent_encoders else None
        inputs=models.WIDTH+(feature_count if self.config.recurrent_observation_skip else 0)
        return Learner(self.base.init(b, inputs),encoder,encoder,self.opt.init(encoder),actor_encoder,
            self.actor_encoder_optimizer.init(actor_encoder) if actor_encoder is not None else None)

    def encode(self,encoder,history):
        return models.recurrent_features(encoder,history,self.config.recurrent_observation_skip)

    def update(self, learner, key, batch):
        if self.config.separate_recurrent_encoders:
            return self.update_separate(learner,key,batch)
        encoded = self.encode(learner.encoder, batch.features)
        next_encoded = self.encode(learner.target_encoder, batch.next_features)
        converted = batch._replace(features=encoded, next_features=next_encoded)
        inner, metrics = self.base.update(learner.inner, key, converted)
        target, prob = base.bootstrap_distribution(inner, jax.random.fold_in(key, 41), converted, self.config)
        temperature = jnp.exp(inner.log_temperatures)

        def loss(encoder):
            features = self.encode(encoder, batch.features)
            current = batch._replace(features=features)
            value_loss = base._distribution_loss(base._taken(inner.critics,current,self.config.critic_layer_norm),target,prob,self.config)
            control, log_c, probability, log_d = base._sample_all(
                inner.actor, jax.random.fold_in(key, 43), features, batch.mask)
            q = base._critics(inner.critics,features,batch.mask,control,self.config.critic_layer_norm).mean(axis=-1).min(axis=0)
            actor_loss = base.actor_objective(probability,log_d,log_c,q,temperature,self.config.categorical_actor_update)
            return value_loss+actor_loss

        value, gradients = jax.value_and_grad(loss)(learner.encoder)
        changes, opt_state = self.opt.update(gradients, learner.encoder_opt)
        encoder = optax.apply_updates(learner.encoder, changes)
        targets = optax.incremental_update(encoder, learner.target_encoder, self.config.target_rate)
        return Learner(inner, encoder, targets, opt_state), jnp.concatenate((metrics, value[None]))

    def update_separate(self,learner,key,batch):
        """Actor and critic memories receive their respective objectives only."""
        critic_features=self.encode(learner.encoder,batch.features)
        critic_next=self.encode(learner.target_encoder,batch.next_features)
        actor_features=self.encode(learner.actor_encoder,batch.features)
        actor_next=self.encode(learner.actor_encoder,batch.next_features)
        converted=batch._replace(features=critic_features,next_features=critic_next)
        inner,metrics=self.base.update(learner.inner,key,converted,
            actor_features=actor_features,next_actor_features=actor_next)
        target,prob=base.bootstrap_distribution(inner,jax.random.fold_in(key,41),converted,self.config,actor_next)
        def critic_loss(encoder):
            current=batch._replace(features=self.encode(encoder,batch.features))
            return base._distribution_loss(base._taken(inner.critics,current,self.config.critic_layer_norm),target,prob,self.config)
        value_c,gradient=jax.value_and_grad(critic_loss)(learner.encoder)
        changes,critic_state=self.opt.update(gradient,learner.encoder_opt)
        encoder=optax.apply_updates(learner.encoder,changes)
        frozen_features=jax.lax.stop_gradient(self.encode(encoder,batch.features))
        temperatures=jnp.exp(inner.log_temperatures)
        def actor_loss(actor_encoder):
            encoded=self.encode(actor_encoder,batch.features)
            control,log_c,p,log_d=base._sample_all(inner.actor,jax.random.fold_in(key,43),encoded,batch.mask)
            values=base._critics(inner.critics,frozen_features,batch.mask,control,self.config.critic_layer_norm)
            selected=jnp.where((values[0].mean(-1)<=values[1].mean(-1))[...,None],values[0],values[1])
            score=base._risk_score(jnp.sort(selected,axis=-1),self.config)
            return base.actor_objective(p,log_d,log_c,score,temperatures,self.config.categorical_actor_update)
        value_a,gradient=jax.value_and_grad(actor_loss)(learner.actor_encoder)
        changes,actor_state=self.actor_encoder_optimizer.update(gradient,learner.actor_encoder_opt)
        actor_encoder=optax.apply_updates(learner.actor_encoder,changes)
        targets=optax.incremental_update(encoder,learner.target_encoder,self.config.target_rate)
        return Learner(inner,encoder,targets,critic_state,actor_encoder,actor_state),jnp.concatenate((metrics,(value_c+value_a)[None]))
