import tensorflow as tf
import gym
import highway_env

from ReplayBuffer import Buffer


class EnvironmentInteractor:

    def __init__(self, env_config: dict, buffer: Buffer, environment_name: str = "highway-fast-v0"):
        self.env_config = env_config

        self.env = gym.make(environment_name)
        self.env.configure(self.env_config)

        self.buffer = buffer
        # Save sizes of the stupid tensors
        self.data_spec = self.buffer.obtain_buffer_specs()

    def create_trajectories(self, iterations):
        state = self.env.reset()

        for _ in range(iterations):
            action = self.env.action_space.sample()

            next_state, reward, done, _ = self.env.step(action)

            self.buffer.add((
                tf.cast(tf.constant(state, shape=self.data_spec[0].shape.as_list()), tf.float32),
                tf.cast(tf.constant(next_state, shape=self.data_spec[1].shape.as_list()), tf.float32),
                tf.cast(tf.constant(action, shape=self.data_spec[2].shape.as_list()), tf.float32),
                tf.cast(tf.constant(reward, shape=self.data_spec[3].shape.as_list()), tf.float32),
                tf.cast(tf.constant(1 - done, shape=self.data_spec[4].shape.as_list()), tf.float32)
            ))

            state = next_state

            if done:
                state = self.env.reset()

    def __del__(self):
        self.env.close()
