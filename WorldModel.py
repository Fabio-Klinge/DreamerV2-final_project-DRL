import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, Reshape, BatchNormalization, GRUCell, MaxPooling2D, Flatten, RNN
import tensorflow_probability as tfp

from RSSM import RSSM, RSSMState
from Parameters import *

class WorldModel:

    def __init__(self) -> None:
        super().__init__()

        self.encoder: tf.keras.Model = self.create_encoder()
        self.decoder: tf.keras.Model = self.create_decoder()
        self.reward_model: tf.keras.Model = self.create_reward_predictor()
        self.discount_model: tf.keras.Model = self.create_discount_predictor()
        self.actor: tf.keras.Model = self.create_actor()
        self.critic: tf.keras.Model = self.create_critic()
        self.target_critic: tf.keras.Model = tf.keras.models.clone_model(self.critic)

        self.rssm: RSSM = RSSM()

        self.models = (self.encoder,
                       self.decoder,
                       self.reward_model,
                       self.discount_model)

    def create_encoder(self, input_size: tuple=image_shape, output_size: int=hidden_unit_size):
        # Third dimension might be obsolete
        encoder_input = tf.keras.Input(shape=input_size)
        x = Conv2D(16, (3, 3), activation="elu", padding="same")(encoder_input)  # 16 layers of filtered 192x48 features
        x = MaxPooling2D((2, 2), padding="same")(x)  # 64 / 96x24
        x = Conv2D(32, (3, 3), activation="elu", padding="same")(x)  # 64 / 96x24
        x = MaxPooling2D((2, 2), padding="same")(x)  # 64 / 96x24
        x = Conv2D(64, (3, 3), activation="elu", padding="same")(x)  # 64 / 48x12
        x = MaxPooling2D((2, 2), padding="same")(x)  # 64 / 48x12
        x = GlobalAveragePooling2D()(x)  # 64
        encoder_output = Dense(output_size, activation="linear")(x)

        encoder = tf.keras.Model(encoder_input, encoder_output, name="Encoder")

        return encoder

    # Input size = 1024(z:32x32) + 200(size of hidden state)
    # Output size = game frame
    def create_decoder(
            self,
            input_size: tuple=stochastic_state_size + hidden_unit_size,
            output_size: tuple=image_shape
    ):
        # Third dimension might be obsolete
        decoder_input = tf.keras.Input(shape=input_size)
        # TODO WIE SCHLIMM IST EIN MLP HIER?
        x = Dense(256, activation="elu")(decoder_input)
        x = Reshape((32, 8, 1))(x)
        # TODO Check whether correct reshape happens
        # tf.debugging.assert_equal(x)
        x = Conv2DTranspose(16, (3, 3), strides=2, activation="elu", padding="same")(x)
        #x = BatchNormalization()(x)
        x = Conv2DTranspose(1, (3, 3), strides=2, activation="linear", padding="same")(x)
        # x = Conv2DTranspose(1, (3, 3), strides=2, activation="elu", padding="same")(x)
        x = Flatten()(x)
        # Might needs shape as Tensor  #event_shape=output_size

        # decoder_output = tfp.layers.IndependentNormal(event_shape=output_size)(x)

        decoder = tf.keras.Model(
            decoder_input,
            x,
            name="Decoder"
        )

        return decoder

        # Input: concatination of h and z

    # Output: float predicting the obtained reward
    def create_reward_predictor(
            self,
            input_size: tuple=hidden_unit_size + stochastic_state_size,
            output_size: int=1
    ):
        reward_predictor_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(reward_predictor_input)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(output_size)(x)
        # Creates indipendent normal distribution
        # Hope is that it learns to output variables over reward space [0,1]
        # reward_predictor_output = tfp.layers.IndependentNormal()(x)

        reward_predictor = tf.keras.Model(
            reward_predictor_input,
            x,
            name="create_reward_predictor"
        )

        return reward_predictor

        # Input: concatination of h and z

    # Output: float predicting the obtained reward
    def create_discount_predictor(
            self,
            input_size: tuple=hidden_unit_size + stochastic_state_size,
            output_size: int=1
    ):
        discount_predictor_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(discount_predictor_input)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(output_size, activation="elu")(x)
        # Create 1 output sampled from bernoulli distribution
        # discount_predictor_output = tfp.layers.IndependentBernoulli()(x)

        discount_predictor = tf.keras.Model(
            discount_predictor_input,
            x,
            name="discount_predictor"
        )

        return discount_predictor

    def create_actor(
            self,
            input_size: tuple=hidden_unit_size + stochastic_state_size,
            output_size: int=action_space_size
    ):
        actor_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(actor_input)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(output_size, activation="linear")(x)

        actor = tf.keras.Model(
            actor_input,
            x,
            name="Actor"
        )

        return actor

    def create_critic(
            self,
            input_size: tuple=hidden_unit_size + stochastic_state_size,
            output_size: int=1
    ):
        critic_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(critic_input)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(output_size, activation="linear")(x)

        actor = tf.keras.Model(
            critic_input,
            x,
            name="Critic"
        )

        return actor

    def compute_actor_critic_loss(self, posterior_rssm_state: RSSMState):

        # TODO At the moment we are using only batches and not batches of sequences#
        batched_posterior_rssm_states = RSSMState.detach(RSSMState.convert_sequences_to_batches(posterior_rssm_state, sequence_length=sequence_length - 1))

        dreamed_rssm_states, dreamed_log_probabilities, dreamed_policy_entropies = self.rssm.dreaming_rollout(horizon, self.actor, batched_posterior_rssm_states)
        dreamed_log_probabilities = tf.reshape(dreamed_log_probabilities, [horizon, batch_size * (sequence_length - 1)])
        dreamed_policy_entropies = tf.reshape(dreamed_policy_entropies, [horizon, batch_size * (sequence_length - 1) ])


        dreamed_hidden_state_h_and_stochastic_state_z = dreamed_rssm_states.get_hidden_state_h_and_stochastic_state_z()

        # TODO HARDCODED SHAPE!
        dreamed_hidden_state_h_and_stochastic_state_z = tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z, (-1, hidden_unit_size + stochastic_state_size))

        self.set_trainable_models(self.models + self.rssm.models + (self.critic,) + (self.target_critic,), False)
        ########################################
        reward_logits = self.reward_model(dreamed_hidden_state_h_and_stochastic_state_z)
        reward_distribution = tfp.distributions.Independent(tfp.distributions.Normal(reward_logits, 1), reinterpreted_batch_ndims=1)
        dreamed_reward = reward_distribution.mean()

        discount_logits = self.discount_model(dreamed_hidden_state_h_and_stochastic_state_z)
        discount_distribution = tfp.distributions.Independent(tfp.distributions.Bernoulli(logits=discount_logits), reinterpreted_batch_ndims=1)
        dreamed_discount = discount_factor * tf.round(discount_distribution.prob(discount_distribution.mean()))

        target_value_logits = self.target_critic(dreamed_hidden_state_h_and_stochastic_state_z)
        target_value_distribution = tfp.distributions.Independent(tfp.distributions.Normal(target_value_logits, 1), reinterpreted_batch_ndims=1)
        dreamed_value = target_value_distribution.mean()
        ########################################
        self.set_trainable_models(self.models + self.rssm.models + (self.critic,) + (self.target_critic,), True)

        actor_loss, discount, lambda_returns = self.actor_loss(dreamed_reward, dreamed_value, dreamed_discount, dreamed_log_probabilities, dreamed_policy_entropies)
        critic_loss = self.critic_loss(dreamed_hidden_state_h_and_stochastic_state_z, discount, lambda_returns)

        return actor_loss, critic_loss

    def actor_loss(self, dreamed_reward, dreamed_value, dreamed_discount, dreamed_log_probabilities, dreamed_policy_entropies, actor_entropy_scale=0.001, lmbda=0.95):
        dreamed_reward = tf.reshape(dreamed_reward, (horizon, -1))
        dreamed_value = tf.reshape(dreamed_value, (horizon, -1))
        dreamed_discount = tf.reshape(dreamed_discount, (horizon, -1))

        lambda_returns = self.compute_return(dreamed_reward[:-1], dreamed_value[:-1], dreamed_discount[:-1], bootstrap=dreamed_value[-1], lmbda=lmbda)
        advantage = tf.stop_gradient(lambda_returns - dreamed_value[:-1])
        objective = dreamed_log_probabilities[1:] * advantage

        discounts = tf.concat([tf.ones_like(dreamed_discount[:1]), dreamed_discount[1:]], 0)
        discount = tf.math.cumprod(discounts[:-1], 0)
        policy_entropy = dreamed_policy_entropies[1:]
        actor_loss = -tf.math.reduce_sum(tf.math.reduce_mean(discount * (objective + actor_entropy_scale * policy_entropy), axis=1))  # TODO correct axis?
        return actor_loss, discount, lambda_returns

    def critic_loss(self, dreamed_hidden_state_h_and_stochastic_state_z, discount, lambda_returns):
        # TODO dreamed_hidden_state_h_and_stochastic_state_z[:-1]
        # TODO Workaround
        dreamed_hidden_state_h_and_stochastic_state_z = tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z, (horizon, batch_size * (sequence_length - 1), -1))
        critic_logits = self.critic(tf.stop_gradient(tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z[:-1], (-1, hidden_unit_size + stochastic_state_size))))
        # TODO Workaround
        critic_logits = tf.reshape(critic_logits, (horizon - 1, batch_size * (sequence_length - 1)))
        critic_distribution = tfp.distributions.Independent(tfp.distributions.Normal(critic_logits, 1))
        critic_loss = -tf.reduce_mean(tf.stop_gradient(tf.reshape(discount, (horizon - 1, -1))) * tf.expand_dims(critic_distribution.log_prob(tf.stop_gradient(lambda_returns)), -1))

        return critic_loss

    def compute_return(self, reward,
                       value,
                       discount,
                       bootstrap,
                       lmbda):

        next_values = tf.concat([value[1:], bootstrap[None]], 0)
        target = reward + discount + next_values * (1 - lmbda)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for timestep in timesteps:
            inp = target[timestep]
            discount_factor = discount[timestep]
            accumulated_reward = inp + discount_factor * lmbda * accumulated_reward
            outputs.append(accumulated_reward)
        returns = tf.reverse(tf.stack(outputs), [0])
        return returns

    def set_trainable_models(self, models, trainable: bool):
        for model in models:
            model.trainable = trainable

    def compute_log_loss(self, distribution, target):
        """
        Computes loss for:
        - Image log loss(Output decoder, frame timestep t)
        - Reward log loss(Output reward network, obtained reward timestep t)
        - Discount log loss(Output of discount network, terminal state timestep t)
        """
        # TODO check whether distribution.log_prob  (target) matches target size
        # histogram von wahrsch. distribution /
        return -tf.math.reduce_mean(distribution.log_prob(target))

    def compute_kl_loss(self, prior_rssm_states, posterior_rssm_states, alpha=0.8):
        """
        alpha: weigh between training the prior toward the representations & regularizing
         the representations towards the prior
        prior: Z
        posterior: Z^
        """
        # TODO Do we need Straight Through Gradients here?
        prior_distribution = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(logits=prior_rssm_states.logits), reinterpreted_batch_ndims=1)
        posterior_distribution = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(logits=posterior_rssm_states.logits), reinterpreted_batch_ndims=1)

        prior_distribution_detached = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(logits=tf.stop_gradient(prior_rssm_states.logits)), reinterpreted_batch_ndims=1)
        posterior_distribution_detached = tfp.distributions.Independent(tfp.distributions.OneHotCategorical(logits=tf.stop_gradient(posterior_rssm_states.logits)), reinterpreted_batch_ndims=1)

        # Loss with KL Balancing
        # TODO check reihenfolge, reduce_mean hat Gradients?!!?
        return alpha * tf.math.reduce_mean(tfp.distributions.kl_divergence(posterior_distribution_detached, prior_distribution)) + (
                    1 - alpha) * tf.math.reduce_mean(tfp.distributions.kl_divergence(posterior_distribution, prior_distribution_detached))


