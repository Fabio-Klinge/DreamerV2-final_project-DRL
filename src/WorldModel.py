import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, Reshape, MaxPooling2D

from Parameters import *
from RSSM import RSSM, RSSMState
from Utils import OneHotDist


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

    def create_encoder(self, input_size: tuple = image_shape, output_size: int = encoding_size):
        """ Create the encoder model. 
        
        :params: input_size: shape of the frame from environment
        :params: output_size: size of the encoding  
        :returns: encoder model
        """

        encoder_input = tf.keras.Input(shape=input_size)
        x = Conv2D(32, (3, 3), activation="elu", padding="same")(encoder_input)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="elu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(128, (3, 3), activation="elu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = GlobalAveragePooling2D()(x)
        encoder_output = Dense(output_size, activation="linear")(x)

        encoder = tf.keras.Model(encoder_input, encoder_output, name="Encoder")

        return encoder

    def create_decoder(
            self,
            input_size: tuple = stochastic_state_size + hidden_unit_size,
            output_size: tuple = image_shape
    ):
        """ 
        Create the decoder model.

        :params: input_size: size of the categorical representations concatenated with the hidden state size
        :params: output_size: size of an image from the environment 
        :returns: decoder model
        """
        decoder_input = tf.keras.Input(shape=input_size)
        x = Dense(1024, activation="elu")(decoder_input)
        x = Reshape((8, 2, 64))(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation="elu", padding="same")(x)
        x = Conv2DTranspose(16, (3, 3), strides=2, activation="elu", padding="same")(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, activation="elu", padding="same")(x)
        x = Conv2DTranspose(8, (3, 3), strides=2, activation="elu", padding="same")(x)
        x = Conv2D(1, (3, 3), strides=1, activation="linear", padding="same")(x)

        decoder = tf.keras.Model(
            decoder_input,
            x,
            name="Decoder"
        )

        return decoder

    def create_reward_predictor(
            self,
            input_size: tuple = hidden_unit_size + stochastic_state_size,
            output_size: int = 1
    ):
        """
        Create the reward predictor model.  
        
        :params: input_size: size of the concatenation of the hidden state and the categorical representation
        :params: output_size: size of the predicted reward
        :returns: reward predictor model
        """
        reward_predictor_input = tf.keras.Input(shape=input_size)
        x = Dense(mlp_hidden_layer_size, activation="elu")(reward_predictor_input)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(mlp_hidden_layer_size, activation="elu")(x)
        x = Dense(output_size)(x)

        reward_predictor = tf.keras.Model(
            reward_predictor_input,
            x,
            name="create_reward_predictor"
        )

        return reward_predictor

        # Input: concatenation of h and z

    # Output: float predicting the obtained reward
    def create_discount_predictor(
            self,
            input_size: tuple = hidden_unit_size + stochastic_state_size,
            output_size: int = 1
    ):
        """
        Create the discount predictor model.

        :params: input_size: size of the concatenation of the hidden state and the categorical representation
        :params: output_size: size of the predicted discount factor 
        :returns: discount predictor model
        """
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
            input_size: tuple = hidden_unit_size + stochastic_state_size,
            output_size: int = action_space_size
    ):
        """
        Create the actor model.

        :params: input_size: size of the concatenation of the hidden state and the categorical representation
        :params: output_size: size of the action space  
        :returns: actor model
        """
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
            input_size: tuple = hidden_unit_size + stochastic_state_size,
            output_size: int = 1
    ):
        """
        Create the critic model.

        :params: input_size: size of the concatenation of the hidden state and the categorical representation.
        :params: output_size: size of the predicted value for the state action pair embedded in the concatenation   
        :returns: critic model
        """
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
        """
        Computes the loss for the actor and critic networks using the posterior state z.

        :params: posterior_rssm_state: posterior state of the RSSM(z) created from the RSSM and the image from the environment
        :returns: actor and critic loss
        """

        batched_posterior_rssm_states = RSSMState.detach(RSSMState.convert_sequences_to_batches(posterior_rssm_state, sequence_length=sequence_length - 1))

        dreamed_rssm_states, dreamed_log_probabilities, dreamed_policy_entropies = self.rssm.dreaming_rollout(horizon, self.actor, batched_posterior_rssm_states)
        dreamed_log_probabilities = tf.reshape(dreamed_log_probabilities, [horizon, -1])
        dreamed_policy_entropies = tf.reshape(dreamed_policy_entropies, [horizon, -1])

        dreamed_hidden_state_h_and_stochastic_state_z = dreamed_rssm_states.get_stochastic_state_z_and_hidden_state_h()

        dreamed_hidden_state_h_and_stochastic_state_z = tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z, (-1, hidden_unit_size + stochastic_state_size))

        self.set_trainable_models(self.models + self.rssm.models + (self.critic,) + (self.target_critic,), False)
        ######################################## # Marks Beginning of Non-Trainable Models
        reward_logits = self.reward_model(dreamed_hidden_state_h_and_stochastic_state_z)
        reward_distribution = tfp.distributions.Independent(tfp.distributions.Normal(reward_logits, 1), reinterpreted_batch_ndims=1)
        dreamed_reward = reward_distribution.mean()

        discount_logits = self.discount_model(dreamed_hidden_state_h_and_stochastic_state_z)
        discount_distribution = tfp.distributions.Independent(tfp.distributions.Bernoulli(logits=discount_logits), reinterpreted_batch_ndims=1)
        dreamed_discount = discount_factor * tf.round(discount_distribution.distribution.probs_parameter())

        target_value_logits = self.target_critic(dreamed_hidden_state_h_and_stochastic_state_z)
        target_value_distribution = tfp.distributions.Independent(tfp.distributions.Normal(target_value_logits, 1), reinterpreted_batch_ndims=1)
        dreamed_value = target_value_distribution.mean()
        ######################################## # Marks End of Non-Trainable Models
        self.set_trainable_models(self.models + self.rssm.models + (self.critic,) + (self.target_critic,), True)

        actor_loss, discount, lambda_returns = self.actor_loss(dreamed_reward, dreamed_value, dreamed_discount, dreamed_log_probabilities, dreamed_policy_entropies)
        critic_loss = self.critic_loss(dreamed_hidden_state_h_and_stochastic_state_z, discount, lambda_returns)

        return actor_loss, critic_loss

    def actor_loss(self, dreamed_reward: tf.Tensor, dreamed_value: tf.Tensor, dreamed_discount: tf.Tensor, dreamed_log_probabilities: tf.Tensor, dreamed_policy_entropies: tf.Tensor, actor_entropy_scale: float = 0.001, lmbda: float = 0.95):
        """
        Computes the actor loss with all the dreamed/predicted values.
        """
        dreamed_reward = tf.reshape(dreamed_reward, (horizon, -1))
        dreamed_value = tf.reshape(dreamed_value, (horizon, -1))
        dreamed_discount = tf.reshape(dreamed_discount, (horizon, -1))

        lambda_returns = self.compute_return(dreamed_reward[:-1], dreamed_value[:-1], dreamed_discount[:-1], bootstrap=dreamed_value[-1], lmbda=lmbda)
        advantage = tf.stop_gradient(lambda_returns - dreamed_value[:-1])
        objective = dreamed_log_probabilities[1:] * advantage

        discounts = tf.concat([tf.ones_like(dreamed_discount[:1]), dreamed_discount[1:]], 0)
        discount = tf.math.cumprod(discounts[:-1], 0)
        policy_entropy = dreamed_policy_entropies[1:]
        actor_loss = -tf.math.reduce_sum(tf.math.reduce_mean(discount * (objective + actor_entropy_scale * policy_entropy), axis=1))
        return actor_loss, discount, lambda_returns

    def critic_loss(self, dreamed_hidden_state_h_and_stochastic_state_z: tf.Tensor, discount: tf.Tensor, lambda_returns: tf.Tensor):
        """
        Computes the critic loss
        """
        dreamed_hidden_state_h_and_stochastic_state_z = tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z, (horizon, -1, hidden_unit_size + stochastic_state_size))
        critic_logits = self.critic(tf.stop_gradient(tf.reshape(dreamed_hidden_state_h_and_stochastic_state_z[:-1], (-1, hidden_unit_size + stochastic_state_size))))
        critic_logits = tf.reshape(critic_logits, (horizon - 1, -1))
        critic_distribution = tfp.distributions.Independent(tfp.distributions.Normal(critic_logits, 1))
        critic_loss = -tf.reduce_mean(tf.stop_gradient(tf.reshape(discount, (horizon - 1, -1))) * tf.expand_dims(critic_distribution.log_prob(tf.stop_gradient(lambda_returns)), -1))

        return critic_loss

    def compute_return(self, reward: tf.Tensor,
                       value: tf.Tensor,
                       discount: tf.Tensor,
                       bootstrap: tf.Tensor,
                       lmbda: float):
        """
        Computes the lambda returns
        """
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

    def set_trainable_models(self, models: list, trainable: bool):
        """
        Set the trainable flag of the given models. This is useful for setting multiple models to non-trainable
        """
        for model in models:
            model.trainable = trainable

    def compute_log_loss(self, distribution: tfp.distributions.Distribution, target: tf.Tensor):
        """
        Computes loss for:
        - Image log loss(Output decoder, frame timestep t)
        - Reward log loss(Output reward network, obtained reward timestep t)
        - Discount log loss(Output of discount network, terminal state timestep t)
        """
        return -tf.math.reduce_mean(distribution.log_prob(target))

    def compute_kl_loss(self, prior_rssm_states: RSSMState, posterior_rssm_states: RSSMState, alpha: float = 0.8):
        """
        Computes the KL loss. Formula is given by the original DreamerV2 paper.
        alpha: weigh between training the prior toward the representations & regularizing
         the representations towards the prior
        """
        prior_distribution = tfp.distributions.Independent(OneHotDist(logits=prior_rssm_states.logits), reinterpreted_batch_ndims=1)
        posterior_distribution = tfp.distributions.Independent(OneHotDist(logits=posterior_rssm_states.logits), reinterpreted_batch_ndims=1)

        prior_distribution_detached = tfp.distributions.Independent(OneHotDist(logits=tf.stop_gradient(prior_rssm_states.logits)), reinterpreted_batch_ndims=1)
        posterior_distribution_detached = tfp.distributions.Independent(OneHotDist(logits=tf.stop_gradient(posterior_rssm_states.logits)), reinterpreted_batch_ndims=1)

        # Loss with KL Balancing
        return alpha * tf.math.reduce_mean(tfp.distributions.kl_divergence(tf.stop_gradient(posterior_distribution_detached), prior_distribution)) + (
                1 - alpha) * tf.math.reduce_mean(tfp.distributions.kl_divergence(posterior_distribution, tf.stop_gradient(prior_distribution_detached)))
