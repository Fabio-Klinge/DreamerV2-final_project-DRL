import tensorflow as tf

# Image size
image_shape = (128,32, 1)

# Long term memory of GRU
hidden_unit_size = 200

# Z in paper
stochastic_state_shape = (32,32)
stochastic_state_size = stochastic_state_shape[0] * stochastic_state_shape[1]

#
action_size = 1
action_space_size = 5
horizon = 15
discount_factor = 0.995

#
mlp_hidden_layer_size = 100
batch_size = 50
sequence_length = 50

epochs = 2048
target_update_every = 5

optimizer_world_model = tf.keras.optimizers.Adam(0.0002, clipnorm=100.0)
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

# TODO different variable names for network inp/outp sizes


