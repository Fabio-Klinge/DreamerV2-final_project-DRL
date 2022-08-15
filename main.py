import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents

import wandb
wandb.init(settings=wandb.Settings(_disable_stats=True), sync_tensorboard=True)

from RSSM import RSSM, RSSMState
from WorldModel import WorldModel
from ReplayBuffer import Buffer
from Agent import Agent
from Trainer import Trainer
from Parameters import *


buffer = Buffer()
world_model = WorldModel()
environment_interactor = Agent(env_config, buffer, world_model)
trainer = Trainer()

wandb.tensorflow.log(tf.summary)


for episode in range(epochs):
    environment_interactor.create_trajectories(batch_size*sequence_length * 2)
    data = buffer.sample(batch_size=batch_size, prefetch_size=10)


    trainer.train_batch(data, world_model)

    if episode % target_update_every == 0:
        tf_agents.utils.common.soft_variables_update(world_model.critic.variables, world_model.target_critic.variables, tau=1.0)

    # Rewards
    # Checkpointing