import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import highway_env

from RSSM import RSSMState
from ReplayBuffer import Buffer
from WorldModel import WorldModel


class Agent:

    def __init__(self, env_config: dict, buffer: Buffer, world_model: WorldModel, environment_name: str = "highway-fast-v0"):
        self.env_config = env_config

        self.env = gym.make(environment_name)
        self.env.configure(self.env_config)

        self.buffer: Buffer = buffer
        # Save sizes of the stupid tensors
        self.data_spec = self.buffer.obtain_buffer_specs()

        self.world_model: WorldModel = world_model

    def create_trajectories(self, iterations):

        state = self.env.reset()
        previous_action = self.env.action_space.sample()
        done = False
        previous_rssm_state = RSSMState()

        for step in range(iterations):

            action, posterior_rssm_state = self.act(state, previous_action, done, previous_rssm_state)

            next_state, reward, done, _ = self.env.step(action.numpy().item())

            self.buffer.add((
                self.preprocess_data(action, done, next_state, reward, state, step)
            ))

            state = next_state
            previous_action = action
            previous_rssm_state = posterior_rssm_state

            if done:
                state = self.env.reset()
                action = self.env.action_space.sample()
                done = False
                previous_rssm_state = RSSMState()

    def preprocess_data(self, action, done, next_state, reward, state, step):
        return self.preprocess_state(state), \
               self.preprocess_next_state(next_state), \
               self.preprocess_action(action), \
               self.preprocess_reward(reward), \
               self.preprocess_done(done), \
                tf.cast(tf.constant(step, shape=self.data_spec[5].shape.as_list()), tf.float32)

    def preprocess_next_state(self, next_state):
        return tf.cast(tf.constant(next_state, shape=self.data_spec[1].shape.as_list()), tf.float32)

    def preprocess_reward(self, reward):
        return tf.cast(tf.constant(reward, shape=self.data_spec[3].shape.as_list()), tf.float32)

    def preprocess_done(self, done):
        return tf.cast(tf.constant(1 - done, shape=self.data_spec[4].shape.as_list()), tf.float32)

    def preprocess_action(self, action):
        return tf.cast(tf.constant(action, shape=self.data_spec[2].shape.as_list()), tf.float32)

    def preprocess_state(self, state):
        return tf.cast(tf.constant(state, shape=self.data_spec[0].shape.as_list()), tf.float32)

    def __del__(self):
        self.env.close()

    def act(self, state, previous_action, non_terminal, previous_rssm_state: RSSMState):
        state = tf.expand_dims(self.preprocess_state(state), 0)
        previous_action = tf.expand_dims(self.preprocess_action(previous_action), 0)
        non_terminal = tf.expand_dims(self.preprocess_done(non_terminal), 0)

        embed = self.world_model.encoder(state, training=False)
        _, posterior_rssm_state = self.world_model.rssm.observe(embed, previous_action, non_terminal, previous_rssm_state)
        hidden_state_h_and_stochastic_state_z = posterior_rssm_state.get_hidden_state_h_and_stochastic_state_z()
        action_logits = self.world_model.actor(hidden_state_h_and_stochastic_state_z)
        action_distribution = tfp.distributions.OneHotCategorical(logits=action_logits)
        action = action_distribution.sample()
        action = tf.argmax(action, axis=-1)

        return action, posterior_rssm_state