import tensorflow as tf

# Image size
image_shape = (128, 32, 1)
buffer_length = 100000

hidden_unit_size = 400
encoding_size = 512

stochastic_state_shape = (32, 32)
stochastic_state_size = stochastic_state_shape[0] * stochastic_state_shape[1]

action_size = 1
action_space_size = 5
horizon = 10
discount_factor = 0.995

mlp_hidden_layer_size = 400
batch_size = 50
prefetch_size = 10
sequence_length = 20

epochs = 2048
target_update_every = 5
save_models_every = 10
continue_training_from_latest_checkpoint = False
logging_weights = False
use_wandb = True

optimizer_world_model = tf.keras.optimizers.Adam(0.002, clipnorm=100.0, epsilon=1e-05)
optimizer_actor = tf.keras.optimizers.Adam(4e-5, clipnorm=100.0)
optimizer_critic = tf.keras.optimizers.Adam(1e-4, clipnorm=100.0)

env_config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 32),
        "stack_size": 1,
        # weights for RGB conversion
        "weights": [0.01, 0.01, 0.98],
        "scaling": 1.5,
    },
    # was at 2
    "policy_frequency": 1
}
