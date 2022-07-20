from pickletools import optimize
import random
from select import select
from keySimulator import KeySimulater
from neuralNetwork import DQN
from memory import ReplayMemory, Transition
from pynput import keyboard
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import math

import time
import cv2
import mss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simulator = KeySimulater() 
listener = None

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


policy_net = DQN()
target_net = DQN()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)
episode_durations = []
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            #Check how to define the outputs to corrospont to keypresses
            return policy_net(state)
    else:
        return torch.tensor([[random.randrange(9)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory < BATCH_SIZE): return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def callAgent(state):
    action = select_action(state)
    reward = 1
    reward = torch.tensor([reward], device=device)
    #TODO: hier weiter https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



def start(key):
    if hasattr(key, 'char'):
        if key.char == 'p': 
            simulator.pressRandomKeys()


def capture():
        debug = False
        capturing = True
        monitor = {"top": 165, "left": 607, "width": 720, "height": 720}

        with mss.mss() as sct:
            # Part of the screen to capture
            
            while capturing:
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(monitor))
                
                # Display the picture
                #cv2.imshow("OpenCV/Numpy normal", img)

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                callAgent(torch.from_numpy(gray_img))

                # Display the picture in grayscale
                cv2.imshow('OpenCV/Numpy grayscale', gray_img)
                if debug: print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    listener = keyboard.Listener(on_press=start, suppress=False) 
    listener.start()

    capture()
    
    

    
