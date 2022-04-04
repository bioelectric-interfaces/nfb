# import numpy as np
# import matplotlib.pyplot as plt
#
# dims = 1
# step_n = 10000
# step_set = [-1, 0, 1]
# origin = np.zeros((1,dims))
# # Simulate steps in 1D
# step_shape = (step_n,dims)
# steps = np.random.choice(a=step_set, size=step_shape)
# path = np.concatenate([origin, steps]).cumsum(0)
# path = path / path.max()
# # Plot the path
# fig = plt.figure(figsize=(8,4),dpi=200)
# ax = fig.add_subplot(111)
# ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05);
# plt.plot(path,c='blue',alpha=0.5,lw=0.5,ls='-',);
# plt.show()
# pass
#
# # Python code for 1-D random walk.
# import random
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Probability to move up or down
# prob = [0.05, 0.95]
#
# # statically defining the starting position
# start = 0
# positions = [start]
#
# # creating the random points
# rr = np.random.random(10000)
# downp = rr > prob[0]
# upp = rr > prob[1]
#
# for idownp, iupp in zip(downp, upp):
#     down = idownp and positions[-1] > 1
#     up = iupp and positions[-1] < 4
#     positions.append(positions[-1] + (down + up))
#
# # plotting down the graph of the random walk in 1D
# plt.plot(positions)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt


def Randomwalk1D(n):  # n here is the no. of steps that we require
    x = 0
    y = 0
    xposition = [0]  # starting from origin (0,0)
    yposition = [0]
    step_size = 0.025
    upp = 0.5
    downp = 0.5
    for i in range(1, n + 1):
        step = np.random.uniform(0, 1)
        if step <= upp:  # if step is less than 0.5 we move up
            y += step_size  # moving up in u direction
        else: #if step > upp:  # if step is greater than 0.5 we move down
            y += -step_size    # moving down in y direction
        x += 1
        xposition.append(x)
        # yposition.append(y)
        yposition +=  [y * l for l in list(np.ones(20))]
        if 0.5 > yposition[-1] > 0.0:
            upp = 0.5
        elif -0.5 < yposition[-1] < 0.0:
            upp = 0.5
        elif 0.75 > yposition[-1] > 0.5:
            upp = 0.4
        elif -0.75 < yposition[-1] < -0.5:
            upp = 0.6
        elif 0.9 > yposition[-1] > 0.75:
            upp = 0.1
        elif -0.9 < yposition[-1] < -0.75:
            upp = 0.9
        elif yposition[-1] > 0.9:
            upp = 0.0
        elif yposition[-1] < -0.9:
            upp = 1
    return [xposition, yposition]


Randwalk = Randomwalk1D(1000)  # creating an object for the Randomwalk1D class and passing value of n as 100
plt.plot(Randwalk[1], 'r-', label="Randwalk1D")  # 'r-' makes the color of the path red
plt.title("1-D Random Walks")
plt.show()