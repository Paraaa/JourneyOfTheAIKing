import time
import cv2
import mss
import numpy as np

class Screen(): 
    def capture(self):
        counter = 0
        debug = False
        capturing = True
        monitor = {"top": 165, "left": 607, "width": 730, "height": 720}

        with mss.mss() as sct:
            # Part of the screen to capture
            
            while capturing:
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(monitor))
                
                # Display the picture
                #cv2.imshow("OpenCV/Numpy normal", img)

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                # Display the picture in grayscale
                cv2.imshow('OpenCV/Numpy grayscale', gray_img)


                #TODO: Feed this image to the DQN

                if debug: print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break