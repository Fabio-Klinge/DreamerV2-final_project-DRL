from typing import NamedTuple

import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, GRUCell

from Parameters import *
from Utils import OneHotDist


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

    def get_stochastic_state_z_and_hidden_state_h(self):
        stochastic_state_z_and_hidden_state_h = tf.concat([self.stochastic_state_z, self.hidden_rnn_state_h], axis=-1)
        return stochastic_state_z_and_hidden_state_h

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
    """
    Recurrent State-Space Model To create categorical representation of image states from the Environment.
    Consists of: GRU cell creating latent space h, encoder network to create image state embeddings,
        decoder Network to convert the categorical representations into images, 
    """

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
        """       
        Gets concatenation of flattend categorical representation z or z^ and the action a. 
        Creates embedding using a dense layer as input for the Reccurent Neural Network (GRU cell).

        :param: input_size: length of flattend z or z^ 
        :param: output_size: length of the embedding 
        :returns: embedding of z or z^ and a
        """
        state_action_input = tf.keras.Input(shape=input_size)
        state_action_output = Dense(output_size, activation="elu")(state_action_input)

        stochastic_state_action_embedder = tf.keras.Model(
            state_action_input,
            state_action_output,
            name="stochastic_state_action_embedder"
        )

        return stochastic_state_action_embedder

    def create_rnn(
            self,
            input_size: tuple = (hidden_unit_size,),
            output_size: int = hidden_unit_size
    ):
        """
        Creates a Recurrent Neural Network with latent space h as hidden state. Used to create categorical representations z or z^.
        Does not use funtional API as layers.GRUCell() was not compatible.

        :param: input_size: length of embedding created by the stochastic_state_action_embedder
        :param: output_size: length of latent state h
        :returns: a GRUCell with latent space h
        """
        return GRUCell(output_size)  # Works and has nearly no points of failure. So we use it over the Legacy Code, down below.

        # # Legacy Code that doesn't work as expected
        # rnn_input = tf.keras.Input(shape=input_size)
        # # rnn_hidden_state_placeholder = tf.keras.Input(shape=(hidden_unit_size,))
        # rnn_output = rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(output_size))(rnn_input)
        #
        # rnn = tf.keras.Model(
        #     rnn_input,
        #     rnn_output,
        #     name="rnn"
        # )
        #
        # return rnn

    def create_prior_stochastic_state_embedder(
            self,
            input_size: tuple = hidden_unit_size,
            output_size: int = stochastic_state_size
    ):
        """
        MLP to create flattend categorical representation z^. 
        Solely created from learned latent space h it is used as state predictions in the A2C model.

        :params: input_size: length of hidden state h 
        :params: output_size size of flattend categorical representation
        :returns: An MLP object to create flattend categorical representation z^


        """
        state_embedder_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(state_embedder_input)
        x = Dense(output_size, activation="elu")(x)
        state_embedder_output = Dense(output_size)(x)

        create_prior_stochastic_state_embedder = tf.keras.Model(
            state_embedder_input,
            state_embedder_output,
            name="create_prior_stochastic_state_embedder"
        )

        return create_prior_stochastic_state_embedder

    def create_posterior_stochastic_state_embedder(
            self,
            input_size: tuple = encoding_size + hidden_unit_size,
            output_size: int = stochastic_state_size
    ):
        """
        MLP to create  categorical representation z (flattend). 
        Created from learned latent space h and the embedded output of the image encoder 
        Used to train state predictions z^ in order to create predictions without environment input.

        :params: input_size: length of hidden state h plus
        :params: output_size size of categorical representation z (flattend)
        :returns: An MLP object to create categorical representation z (flattend)

        """
        state_embedder_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(state_embedder_input)
        x = Dense(output_size, activation="elu")(x)
        state_embedder_output = Dense(output_size)(x)

        create_posterior_stochastic_state_embedder = tf.keras.Model(
            state_embedder_input,
            state_embedder_output,
            name="create_posterior_stochastic_state_embedder"
        )

        return create_posterior_stochastic_state_embedder

    def sample_stochastic_state(self, logits):
        """
        Uses probabilities for each element of class in each category of z or z^ (unflattend) to create
            a One Hot Categorical distribution. Then samples from the distribution.

        :param: logits: probabilities for each element of class in each category of z or z^ (unflattend)
        :returns: sampled categorical representation z or z^
        """

        # Logit Outputs from MLP
        logits = tf.reshape(logits, shape=(-1, *stochastic_state_shape))

        sample = OneHotDist(logits=logits, dtype=tf.float32).sample()

        return tf.reshape(sample, (-1, *stochastic_state_shape))

        # # OneHot distribution over logits
        # logits_distribution = tfp.distributions.OneHotCategorical(logits=logits)
        # # Sample from OneHot distribution
        # sample = tf.cast(logits_distribution.sample(), tf.float32)
        # sample = sample + tf.expand_dims(logits_distribution.prob(sample) - tf.stop_gradient(logits_distribution.prob(sample)), -1)
        #
        # return tf.reshape(sample, (-1, *stochastic_state_shape))

    def dream(self, previous_rssm_state: RSSMState, previous_action: tf.Tensor, non_terminal: tf.Tensor = tf.constant(1.0)):
        """
        Creates a dreamed categorical representation z^ by feeding the previous z or z^, action and flipped terminal to the model. 

        :param: previous_rssm_state: RSSMState object containing previous z or z^
        :param: previous_action: action taken in previous step
        :param: non_terminal: 1 if not terminal, 0.0 if terminal
        :returns: categorical representation z^

        """
        stochastic_state_z = tf.reshape(previous_rssm_state.stochastic_state_z, (-1, stochastic_state_size))
        # Embedding of concatenation prior z and action (t-1)
        state_action_embedding = self.state_action_embedder(tf.concat([stochastic_state_z * non_terminal, previous_action], axis=-1))

        # Create h from GRU with old h (t-1) and the embedding
        state_action_embedding = tf.reshape(state_action_embedding, shape=(-1, hidden_unit_size))
        # previous_rssm_state.hidden_rnn_state = tf.reshape(previous_rssm_state.hidden_rnn_state, shape=(-1, 200))

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
        Output a rollout of z^ of length horizon as "dream" for the actor model.

        :param: horizon: length of rollout
        :param: actor: actor model
        :param: previous_rssm_state: RSSMState object containing previous z or z^
        :returns: rollout of z^
        """
        rssm_state = previous_rssm_state

        next_rssm_states = []
        action_entropies = []
        dream_log_probabilities = []
        for timestep in range(horizon):
            action_logits = actor(tf.stop_gradient(rssm_state.get_stochastic_state_z_and_hidden_state_h()))
            action_distribution = tfp.distributions.OneHotCategorical(logits=action_logits)
            action = action_distribution.sample()

            rssm_state = self.dream(rssm_state, tf.expand_dims(tf.cast(tf.argmax(action, axis=-1), tf.float32), -1))
            next_rssm_states.append(rssm_state)
            action_entropies.append(action_distribution.entropy())
            dream_log_probabilities.append(action_distribution.log_prob(tf.round(tf.stop_gradient(action))))

        next_rssm_states = RSSMState.from_list(next_rssm_states)
        dream_log_probabilities = tf.stack(dream_log_probabilities, 0)
        action_entropies = tf.stack(action_entropies, 0)

        return next_rssm_states, dream_log_probabilities, action_entropies

    def observe(self, encoded_state: tf.Tensor, previous_action: tf.Tensor, previous_non_terminal: tf.Tensor, previous_rssm_state: RSSMState):
        """
        Creates a dreamed categorical representation z^ by feeding the previous z or z^, action and flipped terminal to the model. 

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
        """

        :param encoded_states:
        :param actions:
        :param non_terminals:
        :param previous_rssm_state:
        :return:
        """
        prior_rssm_states = []
        posterior_rssm_states = []

        for sequence_index in range(sequence_length):
            encoded_state, action, non_terminal = encoded_states[:, sequence_index], actions[:, sequence_index], non_terminals[:, sequence_index]

            # 0 if terminal state is reached
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
