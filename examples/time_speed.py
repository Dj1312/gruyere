import sys
sys.path.append('..')

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from gruyere.brushes import notched_square_brush, show_mask
from gruyere.conditional_generator import generator


my_brush = notched_square_brush(5, 1)
N_val = 10 * np.arange(10)

timer = []

for idx, n in enumerate(N_val):
    t0 = perf_counter()
    reward = 2 * np.random.normal(size=(n, n))
    final_des = generator(reward, my_brush)
    t1 = perf_counter()
    timer.append(t1 - t0)
    print(timer, N_val[:idx+1])
    fig, ax = plt.subplots(1,2)
    ax[0].plot(N_val[:idx+1], timer)
    ax[1].plot(N_val[:idx+1], np.log(timer))
    plt.savefig('test_timer.png')

print('Done!')