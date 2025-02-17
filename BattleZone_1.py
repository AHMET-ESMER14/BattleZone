import gym
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from tensorflow.keras.layers import (Input, Flatten, Dense, Concatenate, Multiply, MaxPooling2D, Conv2D)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop

battlezone_envs = ['ALE/BattleZone-v5']

env_number = 0
env = gym.make(battlezone_envs[env_number], render_mode="human")

env.reset()


for _ in range(2000):
    env.render()
    obs, rew, done, _, info = env.step(env.action_space.sample())
    time.sleep(0.02)
    if done:
        env.reset()

env.close()
