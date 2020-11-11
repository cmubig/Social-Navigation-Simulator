import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color=[0.8500, 0.3250, 0.0980], label='The red data')

plt.legend(handles=[red_patch,red_patch,red_patch,red_patch,red_patch])

plt.show()
