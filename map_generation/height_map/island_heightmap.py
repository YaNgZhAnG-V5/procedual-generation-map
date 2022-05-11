import numpy as np
from height_map import HeightMap
from map_generation.utils import *


class IslandHeightMap(HeightMap):
    def get_heightmap(self):
        self.erodability[:] = np.exp(
            np.random.normal(0, 4))
        theta = np.random.random() * 2 * np.pi
        dvxs = 0.5 * (self.vxs + self.dvxs)
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = 0.001 * (1 - distance(self.vxs, 0.5))
        manhattan = np.max(np.abs(self.vxs - 0.5), 1)
        xs = x[(manhattan < 0.3) & (np.abs(y) < 0.1)]
        n = np.random.randint(2, 6)
        for i, u in enumerate(np.linspace(xs.min(), xs.max(), n) - 0.05):
            v = np.random.normal(0, 0.05)
            d = ((x - u) ** 2 + (y - v) ** 2) ** 0.5
            eruption = np.maximum(1 + 0.2 * i - d / 0.15, 0) ** 2
            print("Erupting", self.vxs[np.argmax(eruption), :])
            self.elevation[:-1] = np.maximum(
                self.elevation[:-1], eruption)
            self.do_erosion(20, 0.005)
            self.raise_sealevel(80)

        self.do_erosion(40, 0.005)

        maxheight = 30 * (0.46 - manhattan)
        self.elevation[:-1] = np.minimum(maxheight, self.elevation[:-1])

        sealevel = np.random.randint(85, 92)
        self.raise_sealevel(sealevel)
        self.clean_coast()
        self.finalize()
