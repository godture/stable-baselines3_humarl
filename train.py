import gym
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import HumanoidMirrorActWrapper, HumanoidMirrorObsWrapper

# env = Monitor(HumanoidMirrorObsWrapper(HumanoidMirrorActWrapper(gym.make("Humanoid-v3"))))
env = Monitor(gym.make("Walker2d-v3"))
env = VecFrameStack(DummyVecEnv([lambda: env]), n_stack= )
dir_log = "log/walker/multi/"
name_run = 'test'
net_arch = [256, 256]
seed = 
reset_lazy_freq = None
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
        seed=seed, policy_kwargs={"net_arch": net_arch}, tensorboard_log=os.path.join(dir_log, "tb/"),
        reset_lazy_freq=reset_lazy_freq)

checkpoint_callback = CheckpointCallback(save_freq=60000, save_path=os.path.join(dir_log, "checkpoint/", name_run), name_prefix='model')

model.learn(total_timesteps=10000000, callback=checkpoint_callback, tb_log_name=name_run, reset_num_timesteps=False)