import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, Reshape, BatchNormalization, GRUCell, MaxPooling2D, Flatten, RNN
import tensorflow_probability as tfp

from typing import NamedTuple

from Parameters import *


class RSSMState(NamedTuple):
    logits: tf.Tensor = tf.zeros(shape=(stochastic_state_size,))
    stochastic_state_z: tf.Tensor = tf.zeros(shape=(stochastic_state_size,))
    hidden_rnn_state_h: tf.Tensor = tf.zeros(shape=(hidden_unit_size,))

    @classmethod
    def from_list(cls, rssm_states):
        logits = tf.stack([rssm_state.logits for rssm_state in rssm_states], axis=1)
        stochastic_state_z = tf.stack([rssm_state.stochastic_state_z for rssm_state in rssm_states], axis=1)
        hidden_rnn_state = tf.stack([rssm_state.hidden_rnn_state_h for rssm_state in rssm_states], axis=1)

        return cls(logits, stochastic_state_z, hidden_rnn_state)

    def get_hidden_state_h_and_stochastic_state_z(self):
        hidden_state_h_and_stochastic_state_z = tf.concat([self.stochastic_state_z, self.hidden_rnn_state_h], axis=-1)
        return hidden_state_h_and_stochastic_state_z

    @classmethod
    def detach(cls, rssm_state):
        return cls(tf.stop_gradient(rssm_state.logits), tf.stop_gradient(rssm_state.stochastic_state_z), tf.stop_gradient(rssm_state.hidden_rnn_state_h))

    @classmethod
    def convert_sequences_to_batches(cls, rssm_state, sequence_length):
        logits = cls.convert_sequence_to_batch(rssm_state.logits[:sequence_length])
        stochastic_state_z = cls.convert_sequence_to_batch(rssm_state.stochastic_state_z[:sequence_length])
        hidden_rnn_state = cls.convert_sequence_to_batch(rssm_state.hidden_rnn_state_h[:sequence_length])

        return cls(logits, stochastic_state_z, hidden_rnn_state)

    @classmethod
    def convert_sequence_to_batch(cls, sequence):
        batch = tf.reshape(sequence, (sequence.shape[0] * sequence.shape[1], *sequence.shape[2:]))
        return batch


class RSSM:

    def __init__(self) -> None:
        super().__init__()

        self.state_action_embedder: tf.keras.Model = self.create_stochastic_state_action_embedder()
        self.rnn: tf.keras.layers.Layer = self.create_rnn()
        self.prior_model: tf.keras.Model = self.create_prior_stochastic_state_embedder()
        self.posterior_model: tf.keras.Model = self.create_posterior_stochastic_state_embedder()

        self.models = (self.state_action_embedder,
                       self.rnn,
                       self.prior_model,
                       self.posterior_model)

    def create_stochastic_state_action_embedder(
            self,
            input_size: tuple = (stochastic_state_size + action_size,),
            output_size: int = hidden_unit_size
    ):
        state_action_input = tf.keras.Input(shape=input_size)
        state_action_output = Dense(output_size, activation="elu")(state_action_input)

        stochastic_state_action_embedder = tf.keras.Model(
            state_action_input,
            state_action_output,
            name="stochastic_state_action_embedder"
        )

        return stochastic_state_action_embedder

    # Contains GRU cell
    def create_rnn(
            self,
            input_size: tuple = (hidden_unit_size,),
            output_size: int = hidden_unit_size
    ):
        return GRUCell(output_size)

        rnn_input = tf.keras.Input(shape=input_size)
        # rnn_hidden_state_placeholder = tf.keras.Input(shape=(hidden_unit_size,))
        rnn_output = rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(output_size))(rnn_input)

        rnn = tf.keras.Model(
            rnn_input,
            rnn_output,
            name="rnn"
        )

        return rnn

    # Z^ in paper
    def create_prior_stochastic_state_embedder(
            self,
            input_size: tuple = hidden_unit_size,
            output_size: int = stochastic_state_size
    ):
        state_embedder_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(state_embedder_input)
        # Activation function removed
        state_embedder_output = Dense(output_size)(x)

        create_prior_stochastic_state_embedder = tf.keras.Model(
            state_embedder_input,
            state_embedder_output,
            name="create_prior_stochastic_state_embedder"
        )

        return create_prior_stochastic_state_embedder

    # Z in paper
    # Input size = concatenated output of RNN with output of CNN
    def create_posterior_stochastic_state_embedder(
            self,
            input_size: tuple = hidden_unit_size + hidden_unit_size,
            output_size: int = stochastic_state_size
    ):
        state_embedder_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(state_embedder_input)
        # Activation function removed
        state_embedder_output = Dense(output_size)(x)

        create_posterior_stochastic_state_embedder = tf.keras.Model(
            state_embedder_input,
            state_embedder_output,
            name="create_posterior_stochastic_state_embedder"
        )

        return create_posterior_stochastic_state_embedder

    def sample_stochastic_state(self, logits):
        """
        Gets probabilities for each element of class in each category.
        Used to generate embeddings from logits.
        """

        # Logit Outputs from MLP
        logits = tf.reshape(logits, shape=(-1, *stochastic_state_shape))
        # OneHot distribution over logits
        logits_distribution = tfp.distributions.OneHotCategorical(logits=logits)
        # Sample from OneHot distribution
        sample = tf.cast(logits_distribution.sample(), tf.float32)
        # TODO observe logits_distribution.prob(sample) after few iterations
        # TODO Remove tf.expand_dims - shouldn't be necessary
        sample = sample + tf.expand_dims(logits_distribution.prob(sample) - tf.stop_gradient(logits_distribution.prob(sample)), -1)

        return tf.reshape(sample, (-1, *stochastic_state_shape))

    def dream(self, previous_rssm_state: RSSMState, previous_action: tf.Tensor, non_terminal: tf.Tensor = tf.constant(1.0)):
        """
        Creates Z^
        """
        # TODO ÄNDERN
        stochastic_state_z = tf.reshape(previous_rssm_state.stochastic_state_z, (-1, stochastic_state_size))
        # Embedding of concatenation prior z and action (t-1)
        # TODO Does it work as intended?
        state_action_embedding = self.state_action_embedder(tf.concat([stochastic_state_z * non_terminal, previous_action], axis=-1))

        # TODO Remove Squeeze
        # Create h from GRU with old h (t-1) and the embedding
        state_action_embedding = tf.reshape(state_action_embedding, shape=(-1, hidden_unit_size))
        # TODO ÄNDERN
        # previous_rssm_state.hidden_rnn_state = tf.reshape(previous_rssm_state.hidden_rnn_state, shape=(-1, 200))
        # TODO Which is the correct output? First or last?
        _, hidden_rnn_state = self.rnn(state_action_embedding, tf.reshape(previous_rssm_state.hidden_rnn_state_h * non_terminal, (-1, hidden_unit_size)))

        # Logits created from h (with MLP) to create Z^
        prior_logits = self.prior_model(hidden_rnn_state)
        # Create Z^
        prior_stochastic_state_z = self.sample_stochastic_state(prior_logits)
        # Save logits for Z^, Z^ and h
        prior_rssm_state = RSSMState(prior_logits, tf.reshape(prior_stochastic_state_z, (-1, stochastic_state_size)), hidden_rnn_state)

        return prior_rssm_state

    def dreaming_rollout(self, horizon: int, actor: tf.keras.Model, previous_rssm_state: RSSMState):
        """
        Rollout only Z
        """
        rssm_state = previous_rssm_state

        next_rssm_states = []
        action_entropies = []
        dream_log_probabilities = []
        for timestep in range(horizon):
            action_logits = actor(tf.stop_gradient(rssm_state.get_hidden_state_h_and_stochastic_state_z()))
            action_distribution = tfp.distributions.OneHotCategorical(logits=action_logits)
            action = action_distribution.sample()

            rssm_state = self.dream(rssm_state, tf.expand_dims(tf.cast(tf.argmax(action, axis=-1), tf.float32), -1))
            next_rssm_states.append(rssm_state)
            # TODO is this correct? only entropy of action?
            action_entropies.append(action_distribution.entropy())
            dream_log_probabilities.append(action_distribution.log_prob(tf.round(tf.stop_gradient(action))))

        next_rssm_states = RSSMState.from_list(next_rssm_states)
        dream_log_probabilities = tf.stack(dream_log_probabilities, 0)
        action_entropies = tf.stack(action_entropies, 0)

        return next_rssm_states, dream_log_probabilities, action_entropies

    def observe(self, encoded_state: tf.Tensor, previous_action: tf.Tensor, previous_non_terminal: tf.Tensor, previous_rssm_state: RSSMState):
        """
        Creates Z' and Z
        """
        # Obtain Z^
        prior_rssm_state = self.dream(previous_rssm_state, previous_action, previous_non_terminal)

        # concatenates h and the output of our CNN (encoded input frame X)
        encoded_state_and_hidden_state = tf.concat([prior_rssm_state.hidden_rnn_state_h, encoded_state], axis=-1)

        # Logits created from concat of h and encoded frame X (with MLP) to create Z
        posterior_logits = self.posterior_model(encoded_state_and_hidden_state)
        # Create Z
        posterior_stochastic_state_z = self.sample_stochastic_state(posterior_logits)
        # Saves logits for Z, Z, and h
        posterior_rssm_state = RSSMState(posterior_logits, tf.reshape(posterior_stochastic_state_z, (-1, stochastic_state_size)), prior_rssm_state.hidden_rnn_state_h)

        return prior_rssm_state, posterior_rssm_state

    def observing_rollout(self, encoded_states: tf.Tensor, actions: tf.Tensor, non_terminals: tf.Tensor, previous_rssm_state: RSSMState):
        prior_rssm_states = []
        posterior_rssm_states = []

        for sequence_index in range(sequence_length):
            encoded_state, action, non_terminal = encoded_states[:, sequence_index], actions[:, sequence_index], non_terminals[:, sequence_index]

            # ?? 0 if terminal state is reached
            previous_action = action * non_terminal
            # Z^, Z
            prior_rssm_state, posterior_rssm_state = self.observe(encoded_state, previous_action, non_terminal, previous_rssm_state)

            # Save Z^, Z
            prior_rssm_states.append(prior_rssm_state)
            posterior_rssm_states.append(posterior_rssm_state)

            # Z for next iteration
            previous_rssm_state = posterior_rssm_state
        prior_rssm_states = RSSMState.from_list(prior_rssm_states)
        posterior_rssm_states = RSSMState.from_list(posterior_rssm_states)

        return prior_rssm_states, posterior_rssm_states
