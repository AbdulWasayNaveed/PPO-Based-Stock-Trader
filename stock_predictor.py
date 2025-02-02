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

 # Reading Data
df = pd.read_csv('C:/Users/user/Desktop/Stock Prediction Python/gmedata.csv') # Change path according to your file location
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
df.set_index('Date', inplace=True)
df.sort_index()
df.head()
 

# Biliding Enviornment
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env.signal_features

env.action_space
state = env.reset()
while True: 
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done: 
        print("info", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()



model_dir = 'C:/Users/user/Desktop/Stock Prediction Python/PPO'
log_dir ='C:/Users/user/Desktop/Stock Prediction Python/Logs'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)



# Building model
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])
model = PPO('MlpPolicy', env,policy_kwargs=dict(net_arch=[64, 64, 64]), verbose=1,tensorboard_log=log_dir)
Timesteps = 10000
for i in range(1,30): 
    model.learn(total_timesteps=Timesteps,reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{Timesteps*i}")


   

