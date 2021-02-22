import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import matplotlib
matplotlib.rcParams.update({'font.size': 13})

plt_colors = []
plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
plt_colors.append([0.0, 0.4470, 0.7410])  # blue
plt_colors.append([0.4660, 0.6740, 0.1880])  # green
plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
plt_colors.append([0.9290, 0.6940, 0.1250])  # yellow
plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate

filename  = "test_case_0.txt" #"test_case_0.txt" #"STGCNN.txt"#"biwi_hotel.txt" #"biwi_eth_segement.txt"#
#"univ.txt"
#"biwi_eth.txt"
#'test_case_0.txt'
#"crowds_zara01.txt"
#"crowds_zara02.txt"

name = filename.split(".")[0]

file = open(filename, 'rb')

lines=file.readlines()

def color(agent_id):
    return plt_colors[agent_id%len(plt_colors)]

timestamp_list = []
agent_id_list  = []
agent_x_list   = []
agent_y_list   = []

for line in lines:
    timestamp,agent_id,agent_x,agent_y = line.strip().split()
    timestamp_list.append(int(float(timestamp)))
    agent_id_list.append(int(float(agent_id)))
    agent_x_list.append(float(agent_x))
    agent_y_list.append(float(agent_y))

unique_agent_list = np.unique( np.array(agent_id_list) )

timespan = len(np.unique( np.array(timestamp_list) ))

start_time = int(np.min(timestamp_list))
end_time   = int(np.max(timestamp_list))

timestamp_list = np.array(timestamp_list)
temp = timestamp_list[np.where(timestamp_list==10)]



fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(8, 8)

x_min = -1  #-5   #-1 17 for univ   #-5 15 for biwi eth  #-3 17 for zara1  #biwi_hotel -12 8
x_max = 6   #15
y_min = -1  #-5
y_max = 6   #15

plt.axis([x_min,x_max,y_min,y_max]) #for simulator
#plt.axis([0,20,0,20]) #for biwi_eth

ax = plt.gca()
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(x_min, x_max+1, 1)
minor_ticks = np.arange(x_min, x_max+1, 0.2)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.minorticks_on()

# Customize the major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='red'   , alpha=0.2)
# Customize the minor grid
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black' , alpha=0.2)




metadata = dict(title=name, artist='Sam Shum',
                comment='visualization')
writer_mp4 = animation.FFMpegFileWriter(fps=(15), metadata=metadata)
writer_gif = animation.ImageMagickFileWriter(fps=(15), metadata=metadata)


def init():
    return []




def animate(i):
    patches = []
    ax.patches = []
    ax.annotations = []
    

    records = np.where(timestamp_list==(start_time+i*10))[0]
    #print(records)
    for record in records:

        patches.append(ax.add_patch( plt.Circle((agent_x_list[record]+0.2,agent_y_list[record]+0.2),0.2,color=color(agent_id_list[record]) ,linewidth=0.00001      ) ))
        #patches.append(ax.annotate(str(agent_id_list[record]), xy=(agent_x_list[record]+0.4, agent_y_list[record]+0.2), fontsize=10, ha="center"))
        patches.append(ax.legend(["Timestep "+str(timestamp_list[record]).rjust(6)],loc="upper right", prop={'size': 14},handlelength=0.00001, handletextpad=0.00001, markerfirst=False))

        
    print(i)    
    return patches


anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=timespan,
                               interval=200, #ms for each frame
                               blit=True,
                               repeat=False)

anim.save("output/"+str(name)+'.mp4', writer=writer_mp4)
#anim.save("output/"+str(name)+'.gif', writer=writer_gif)
#plt.show()
