import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents

import numpy as np
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

checkpoint_world_model = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_world_model, net=world_model.models + world_model.rssm.models)
checkpoint_actor = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_actor, net=world_model.actor)
checkpoint_critic = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer_critic, net=world_model.critic)

manager_world_model = tf.train.CheckpointManager(checkpoint_world_model, './checkpoints/world_model', max_to_keep=10, keep_checkpoint_every_n_hours=1)
manager_actor = tf.train.CheckpointManager(checkpoint_actor, './checkpoints/actor', max_to_keep=10, keep_checkpoint_every_n_hours=1)
manager_critic = tf.train.CheckpointManager(checkpoint_critic, './checkpoints/critic', max_to_keep=10, keep_checkpoint_every_n_hours=1)

if continue_training_from_latest_checkpoint:
    checkpoint_world_model.restore(manager_world_model.latest_checkpoint)
    checkpoint_actor.restore(manager_actor.latest_checkpoint)
    checkpoint_critic.restore(manager_critic.latest_checkpoint)

wandb.tensorflow.log(tf.summary)


def checkpointing():
    checkpoint_world_model.step.assign_add(1)
    checkpoint_actor.step.assign_add(1)
    checkpoint_critic.step.assign_add(1)
    if int(checkpoint_world_model.step) % save_models_every == 0:
        save_path_world_model = manager_world_model.save()
        save_path_actor = manager_actor.save()
        save_path_critic = manager_critic.save()
        print("Saved checkpoint for step {}: {}, {}, {}".format(int(checkpoint_world_model.step), save_path_world_model, save_path_actor, save_path_critic))


def calculate_mean_episode_length():
    lengths = 0
    for score in scores:
        lengths += len(score)
    lengths /= len(scores)
    return lengths


def log_model_weights_wandb():
    for model in world_model.models + world_model.rssm.models:
        if isinstance(model, tf.keras.layers.GRUCell):
            continue
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2DTranspose):
                # print(layer.get_weights()[0].shape)
                wandb.log({f"weights {model.name}": wandb.Histogram(layer.get_weights()[0])})
                wandb.run.summary.update({f"weights {model.name}": wandb.Histogram(layer.get_weights()[0])})


for episode in range(epochs):
    scores = environment_interactor.create_trajectories(steps=batch_size * sequence_length * 2)
    data = buffer.sample()

    trainer.train_batch(data, world_model)

    checkpointing()

    summed_scores = [sum(score) for score in scores]
    wandb.log({"Score": np.mean(summed_scores)})
    print(f"Episode {episode}: {np.mean(summed_scores)}")

    lengths = calculate_mean_episode_length()
    wandb.log({"Length": lengths})
    print(f"Length: {lengths}")

    if episode % target_update_every == 0:
        tf_agents.utils.common.soft_variables_update(world_model.critic.variables, world_model.target_critic.variables, tau=1.0)

        log_model_weights_wandb()

    # Rewards
    # Checkpointing
