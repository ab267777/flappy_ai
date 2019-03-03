import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque


game_state = game.GameState()
do_nothing = np.zeros(2)
do_nothing[0] = 1
x_t1_colored, r_t, terminal = game_state.frame_step(do_nothing)
print(x_t1_colored,r_t,terminal)

cv2.imshow("img",x_t1_colored)
cv2.waitKey(0)