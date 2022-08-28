import gym
import os
env = gym.make("Humanoid-v3")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

dir_log = "log/humanoid/single/"
name_run = '0003'
net_arch = [256, 256]
seed = 
policy = "HumarlPolicy"

if os.path.isdir(os.path.join(dir_log, "checkpoint/", name_run)) and os.listdir(os.path.join(dir_log, "checkpoint/", name_run)):
    files = os.listdir(os.path.join(dir_log, "checkpoint/", name_run))
    last_checkstep = max([int(file.split("_")[-2]) for file in files if file.split("_")[-3]=='model'])
    path_checkpoint = os.path.join(dir_log, "checkpoint/", name_run, f"model_{last_checkstep}_steps")
    path_buffer = os.path.join(dir_log, "checkpoint/", name_run, f"buffer_{last_checkstep}_steps")
    model = SAC.load(path_checkpoint, env=env)
    model.load_replay_buffer(path_buffer)
else:
    model = SAC(policy, env, verbose=1, learning_starts=10000,
        seed=seed, policy_kwargs={"net_arch": net_arch}, tensorboard_log=os.path.join(dir_log, "tb/"))

checkpoint_callback = CheckpointCallback(save_freq=60000, save_path=os.path.join(dir_log, "checkpoint/", name_run), name_prefix='model')

model.learn(total_timesteps=10000000, callback=checkpoint_callback, tb_log_name=name_run, reset_num_timesteps=False)