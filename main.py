from screen import Screen
from keySimulator import KeySimulater
from neuralNetwork import DQN
from pynput import keyboard

screen = Screen()
simulator = KeySimulater() 

listener = None

def start(key):
    if hasattr(key, 'char'):
        if key.char == 'p': 
            simulator.pressRandomKeys()


if __name__ == '__main__':
    listener = keyboard.Listener(on_press=start, suppress=False) 
    listener.start()

    screen.capture()
    
    

    
