import random
import numpy as np
from matplotlib import pyplot as plt
from map_generation.map import MapGrid
from map_generation.plotter import Plotter

if __name__ == '__main__':
    seed = 2000
    random.seed(seed)
    np.random.seed(seed)
    for i in range(1):
        for mode in ["shore/island", "island", "mountain", "desert"]:

            plt.close('all')
            while True:
                m = MapGrid(mode=mode)
                m.generate_map()
                filename = "./tests/%s-%02d.png" % (m.mode, i)
                plotter = Plotter(m)
                plotter.plot(filename)
                break
