import gym
import os
env = gym.make("Humanoid-v3")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

dir_log = "log/humanoid/local_masac/"
net_arch = [128, 128]

checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=os.path.join(dir_log, "checkpoint/"), name_prefix='model')
model = SAC("HumarlPolicy", env, verbose=1, learning_starts=10000,
  seed=61, policy_kwargs={"net_arch": net_arch}, tensorboard_log=os.join(dir_log, "tb/"))

model.learn(total_timesteps=10000000, callback=checkpoint_callback)