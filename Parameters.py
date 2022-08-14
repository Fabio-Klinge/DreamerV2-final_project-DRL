# Image size
image_shape = (128,32, 1)

# Long term memory of GRU
hidden_unit_size = 200

# Z in paper
stochastic_state_shape = (32,32)
stochastic_state_size = stochastic_state_shape[0] * stochastic_state_shape[1]

#
action_size = 1
horizon = 15
discount_factor = 0.995

#
mlp_hidden_layer_size = 100
batch_size = 50
sequence_length = 50

epochs = 1024

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


