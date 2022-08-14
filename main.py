from functools import reduce
from operator import add

import tensorflow as tf
import tensorflow_probability as tfp

import wandb
wandb.init(settings=wandb.Settings(_disable_stats=True))

from RSSM import RSSM, RSSMState
from WorldModel import WorldModel
from ReplayBuffer import Buffer
from Agent import EnvironmentInteractor
from Parameters import *

optimizer_world_model = tf.keras.optimizers.Adam(0.0002, clipnorm=100.0)
optimizer_actor = tf.keras.optimizers.Adam(4e-5, clipnorm=100.0)
optimizer_critic = tf.keras.optimizers.Adam(1e-4, clipnorm=100.0)

buffer = Buffer()

environment_interactor = EnvironmentInteractor(env_config, buffer)

world_model = WorldModel()


wandb.tensorflow.log(tf.summary)

combined_trainable_variables = reduce(add, [model.trainable_variables for model in (world_model.models + world_model.rssm.models)])

for episode in range(epochs):
    environment_interactor.create_trajectories(500)
    data = buffer.sample(batch_size=50, prefetch_size=70)
    # Sample from buffer
    for sequence in data:
        state, next_state, action, reward, non_terminal = sequence[0]

        with tf.GradientTape() as tape:
            encoded_state = world_model.encoder(state)
            initial_rssm_state = RSSMState()
            prior_rssm_states, posterior_rssm_states = world_model.rssm.observing_rollout(encoded_state, action, non_terminal, initial_rssm_state)
            hidden_state_h_and_stochastic_state_z = tf.concat([posterior_rssm_states.stochastic_state_z, posterior_rssm_states.hidden_rnn_state], axis=-1)

            # TODO ÄNDERN
            hidden_state_h_and_stochastic_state_z = tf.reshape(hidden_state_h_and_stochastic_state_z, (-1, stochastic_state_size + hidden_unit_size))

            decoder_logits = world_model.decoder(hidden_state_h_and_stochastic_state_z)

            # TODO ÄNDERN
            decoder_logits = tf.reshape(decoder_logits, (-1, image_shape[0], image_shape[1], image_shape[2]))

            decoder_distribution = tfp.distributions.Independent(tfp.distributions.Normal(decoder_logits, 1), reinterpreted_batch_ndims=3)
            reward_logits = world_model.reward_model(hidden_state_h_and_stochastic_state_z)
            reward_distribution = tfp.distributions.Independent(tfp.distributions.Normal(reward_logits, 1), reinterpreted_batch_ndims=1)
            discount_logits = world_model.discount_model(hidden_state_h_and_stochastic_state_z)
            discount_distribution = tfp.distributions.Independent(tfp.distributions.Bernoulli(logits=discount_logits), reinterpreted_batch_ndims=1)

            image_log_loss = world_model.compute_log_loss(decoder_distribution, state)
            reward_log_loss = world_model.compute_log_loss(reward_distribution, reward)
            discount_log_loss = world_model.compute_log_loss(discount_distribution, non_terminal)
            kl_loss = world_model.compute_kl_loss(prior_rssm_states, posterior_rssm_states)

            loss = 0.1 * image_log_loss + reward_log_loss + discount_log_loss + 5.0 * kl_loss

        predicted_state = wandb.Image((decoder_distribution.sample(1)[0][0] + 1) * 128)
        real_state = wandb.Image((state[0] + 1) * 128)
        wandb.log({"predicted_state": predicted_state, "real_state": real_state})
        print(f"Image Log Loss: {image_log_loss} Reward Log Loss: {reward_log_loss} Discount Log Loss {discount_log_loss} KL Loss {kl_loss}")
        wandb.log({"Image Log Loss": image_log_loss, "Reward Log Loss": reward_log_loss, "Discount Log Loss": discount_log_loss, "KL Loss": kl_loss, "Loss": loss})

        # TODO maybe in Gradienttape??
        gradients = tape.gradient(loss, combined_trainable_variables)

        optimizer_world_model.apply_gradients(zip(gradients, combined_trainable_variables))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            actor_loss, critic_loss = world_model.compute_actor_critic_loss(posterior_rssm_states)

        print(f"Actor Loss: {actor_loss} Critic Loss: {critic_loss}")
        wandb.log({"Actor Loss": actor_loss, "Critic Loss": critic_loss})

        gradients_actor = tape1.gradient(actor_loss, world_model.actor.trainable_variables)
        gradients_critic = tape2.gradient(critic_loss, world_model.critic.trainable_variables)

        optimizer_actor.apply_gradients(zip(gradients_actor, world_model.actor.trainable_variables))
        optimizer_critic.apply_gradients(zip(gradients_critic, world_model.critic.trainable_variables))
