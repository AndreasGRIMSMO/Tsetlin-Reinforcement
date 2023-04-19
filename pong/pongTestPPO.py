import cv2
import gym
import random
from matplotlib import animation
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
np.set_printoptions(threshold=sys.maxsize)
from gym import spaces
import atariwrappers
from gym.wrappers import AtariPreprocessing

def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='imagemagick')


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(float).ravel()

# Frame list collector
frames = []
STEPS = 300

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
# initializing our environment
env = gym.make("PongNoFrameskip-v4")
env = VecFrameStack(env, n_stack=4)
checkpoint = load_from_hub(
    repo_id="sb3/ppo-PongNoFrameskip-v4",
    filename="ppo-PongNoFrameskip-v4.zip",
)
model = PPO.load(checkpoint)

# beginning of an episode
observation = env.reset()
_states = None

# main loop
for i in range(STEPS):

    # choose random action
    action, _state = model.predict(observation, state=_states)

    # run one step
    observation, reward, done, info = env.step(action)

    # preprocess the observation, set input as difference between images
    cur_input = prepro(observation)
    x = cur_input
    frames.append(x.reshape(80, 80).astype('uint8'))  # collecting observation

    # if episode is over, reset to beginning
    if done:
        observation = env.reset()
        frames.append(x.reshape(80, 80).astype('uint8'))  # collecting observation

frames_bis = []
for x in frames[150:]:
    frames_bis.append(Image.fromarray(x))

#save_frames_as_gif(frames_bis, filename='pong-random-300-steps-black-and-white.gif')
