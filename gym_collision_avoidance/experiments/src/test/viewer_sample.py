import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

##file = open('.txt', 'rb')
##lines=file.readlines()
##
##timestamp_list = []
##agent_id_list  = []
##agent_x_list   = []
##agent_y_list   = []
##
##for line in lines:
##    timestamp,agent_id,agent_x,agent_y = line.split(" ")
##    timestamp_list.append(timestamp)
##    agent_id_list.append(agent_id)
##    agent_x_list.append(agent_x)
##    agent_y_list.append(agent_y)



fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(8, 8)

ax = plt.axes(xlim=(0, 100), ylim=(0, 100))
ax.minorticks_on()

# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

enemy = plt.Circle((10, -10), 0.75, fc='r')
agent = plt.Circle((10, -10), 0.75, fc='b')


def init():
    enemy.center = (5, 5)
    agent.center = (5, 5)
    ax.add_patch(agent)
    ax.add_patch(enemy)

    return []


def animationManage(i,agent,enemy):
    animateCos(i,enemy)
    animateLine(i,agent)
    return []


def animateLine(i, patch):
    x, y = patch.center
    x += 0.25
    y += 0.25
    patch.center = (x, y)
    return patch,


def animateCos(i, patch):
    x, y = patch.center
    x += 0.2
    y = 50 + 30 * np.cos(np.radians(i))
    patch.center = (x, y)
    return patch,

anim = animation.FuncAnimation(fig, animationManage,
                               init_func=init,
                               frames=360,
                               fargs=(agent,enemy,),
                               interval=20,
                               blit=True,
                               repeat=True)


plt.show()
