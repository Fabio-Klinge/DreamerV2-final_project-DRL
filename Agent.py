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
        '''
        Create environment, Experience Replay Buffer and World_model objects.

        :param: env_config: includes information to adjust environment settings like e.g. state size
        :param: buffer: Buffer Object to use as Experience Replay buffer
        :param: world_model: World model object containing RSSM, Reward-, Image- and Discount predictor
        :param: environment_name: selects the environment
        '''
        self.env_config = env_config

        self.env = gym.make(environment_name)
        self.env.configure(self.env_config)

        self.buffer: Buffer = buffer
        # Save sizes of tensors in replay buffer
        self.data_spec = self.buffer.obtain_buffer_specs()

        self.world_model: WorldModel = world_model

    def create_trajectories(self, iterations: int):
        '''
        Collect data from environment and save it to Experience Replay Buffer.

        :param: iterations: number of steps taken in the environment
        '''

        state = self.env.reset()
        previous_action = self.env.action_space.sample()
        done = False
        previous_rssm_state = RSSMState()

        for step in range(iterations):

            # Obtain action and embedded environment state
            action, posterior_rssm_state = self.act(state, previous_action, done, previous_rssm_state)

            next_state, reward, done, _ = self.env.step(action.numpy().item())

            self.buffer.add((
                self.preprocess_data(action, done, next_state, reward, state, step)
            ))

            # Update for next iteration
            state = next_state
            previous_action = action
            previous_rssm_state = posterior_rssm_state

            # Terminal state
            if done:
                state = self.env.reset()
                action = self.env.action_space.sample()
                done = False
                previous_rssm_state = RSSMState()

    def preprocess_data(self, action, done, next_state, reward, state, step):
        '''
        Adjusts data to the specific shape and type needed for the Experience Replay Buffer.

        :params: action: action taken
        :params: done: Terminal Boolean
        :params: next_state: New state s' after taking action a in s
        :params: reward: Reward for taking a in s
        :params: state: current state s
        :params: step: 
        :returns: Data adjusted to be added to Buffer
        '''
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
        '''
        End current instance of environment.
        '''
        self.env.close()

    def act(self, state, previous_action, non_terminal, previous_rssm_state: RSSMState):
        '''
        :params: state: Current state s
        :params: previous_action: action taken in s
        :params: non_terminal: Flipped terminal Boolean
        :params: previous_rssm_state: Object containing logits for z from prior iteration
        :returns: action: The sampled action from the stochastic actor output
        :returns: posterior_rssm_state: Object containing logits for z,z^ and h
        '''
        # Create dummy batch dimensions
        state = tf.expand_dims(self.preprocess_state(state), 0)
        previous_action = tf.expand_dims(self.preprocess_action(previous_action), 0)
        non_terminal = tf.expand_dims(self.preprocess_done(non_terminal), 0)

        # Create latent space vector of state with encoder
        embed = self.world_model.encoder(state, training=False)
        # Obtain object containing logits from posterior state z
        _, posterior_rssm_state = self.world_model.rssm.observe(embed, previous_action, non_terminal, previous_rssm_state)
        # Obtain logits from posterior state z
        hidden_state_h_and_stochastic_state_z = posterior_rssm_state.get_hidden_state_h_and_stochastic_state_z()

        # Create distribution over actions and sample from it
        action_logits = self.world_model.actor(hidden_state_h_and_stochastic_state_z)
        action_distribution = tfp.distributions.OneHotCategorical(logits=action_logits)
        action = action_distribution.sample()
        action = tf.argmax(action, axis=-1)

        return action, posterior_rssm_state