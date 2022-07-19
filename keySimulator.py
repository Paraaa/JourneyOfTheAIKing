import random
import time 
from pynput.keyboard import Key,Controller

class KeySimulater():

    delay = 0.05
 
    keyboard = Controller()
    shooting_keys = [
        Key.up, Key.down, Key.left, Key.right
    ]
    moving_keys = [
        'w', 'a', 's', 'd'
    ]
    special_keys = [
        Key.space
    ]

    def pressRandomKeys(self):
        print("Pressing keys")
        
        keys_to_press = []

        keys_to_press.append(random.choice(self.shooting_keys))
        keys_to_press.append(random.choice(self.shooting_keys))
        keys_to_press.append(random.choice(self.moving_keys))
        keys_to_press.append(random.choice(self.moving_keys))
        keys_to_press.append(random.choice(self.special_keys))


        for key in keys_to_press:
            self.keyboard.press(key)
        time.sleep(self.delay)
        for key in keys_to_press:
            self.keyboard.release(key)
        print("Stopped")
        

            
            