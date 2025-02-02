import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Trading Enviornment
import gym
import gym_anytrading
# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
# Data Processing & Plotting
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Loading Model
df = pd.read_csv('C:/Users/user/Desktop/Stock Prediction Python/gmedata.csv') # Change path according to your file location
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
df.set_index('Date', inplace=True)
df.head()
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env.reset()
model_path = "C:/Users/user/Desktop/Stock Prediction Python/PPO/200000.zip"



# Evaluation
model = PPO.load(model_path, env=env)
env = gym.make('stocks-v0', df=df, frame_bound=(110,150), window_size=5)
obs = env.reset()
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
