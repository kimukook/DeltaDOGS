import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

fig = plt.figure()
plt.scatter(self.xE[0, :], self.xE[1, :])
plt.plot([0, 0], [0, 1], c='k')
plt.plot([0, 1], [1, 1], c='k')
plt.plot([1, 1], [1, 0], c='k')
plt.plot([1, 0], [0, 0], c='k')
plt.grid()
plt.yticks(visible=False)
plt.xticks(visible=False)
plt.show()

