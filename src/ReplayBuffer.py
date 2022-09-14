# Neural Network
# Buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from Parameters import *


class Buffer:

    def __init__(
            self,
            batch_size: int = 1,
            buffer_length: int = buffer_length,
            observation_size: tuple = image_shape,
            action_size: int = action_size
    ):
        """
        Create replay buffer

        Buffer size = batch_size * buffer_length

        """
        # Save batch size for other functions of buffer
        # NOT the usual batch size in Deep Learning
        # Batches in Uniform Replay Buffer describe size of input added to the buffer
        self.batch_size = batch_size

        # Tell buffer what data & which size to expect
        self.data_spec = (
            tf.TensorSpec(
                shape=observation_size,
                dtype=tf.dtypes.float32,
                name="Observation"
            ),
            tf.TensorSpec(
                shape=observation_size,
                dtype=tf.dtypes.float32,
                name="Next state"
            ),
            tf.TensorSpec(
                shape=[action_size],
                dtype=tf.dtypes.float32,
                name="Action"
            ),
            tf.TensorSpec(
                # Reward size
                shape=[1, ],
                dtype=tf.dtypes.float32,
                name="Reward"
            ),
            tf.TensorSpec(
                shape=[1, ],
                dtype=tf.dtypes.float32,
                name="Non-Terminal State"
            ),
            tf.TensorSpec(
                shape=[1, ],
                dtype=tf.dtypes.float32,
                name="Index"
            )
        )

        # Create the buffer
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.data_spec, batch_size, buffer_length
        )

    def obtain_buffer_specs(self):
        return self.data_spec

    def add(self, items):
        """
        length of items must be equal to batch size

        items: list or tuple of batched data from (50, 5)

        """
        batched_values = tf.nest.map_structure(
            lambda t: tf.stack([t] * self.batch_size),
            items
        )

        # Add to batch
        self.buffer.add_batch(batched_values)

    def sample(self):
        data = self.buffer.as_dataset(num_steps=sequence_length, single_deterministic_pass=True)

        data = data.map(lambda buffer_content, _: (((buffer_content[0] / 128.) - 1, (buffer_content[1] / 128.) - 1, buffer_content[2], buffer_content[3], buffer_content[4], buffer_content[5]), _))
        data = data.cache()
        data = data.batch(batch_size, drop_remainder=True).prefetch(prefetch_size)
        return data
