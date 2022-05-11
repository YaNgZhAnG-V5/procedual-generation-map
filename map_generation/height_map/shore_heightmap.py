import numpy as np
from map_generation.height_map.height_map import HeightMap
from map_generation.utils import *


class ShoreHeightMap(HeightMap):
    def get_heightmap(self):
        print("Calculating elevations")
        n = self.vxs.shape[0]
        self.elevation = np.zeros(n + 1)
        self.elevation[:-1] = 0.5 + ((self.dvxs - 0.5) * np.random.normal(0, 4, (1, 2))).sum(1)
        self.elevation[:-1] += -4 * (np.random.random() - 0.5) * distance(self.vxs, 0.5)
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2
        print("Edge height:", self.elevation[:-1][self.edge].max())

        along = (((self.dvxs - 0.5) * np.random.normal(0, 2, (1, 2))).sum(1) + np.random.normal(0, 0.5)) * 10
        self.erodability = np.exp(4 * np.arctan(along))

        for i in range(5):
            self.rift()
            self.relax()
        for i in range(5):
            self.relax()
        self.normalize_elevation()

        sealevel = np.random.randint(20, 40)
        self.raise_sealevel(sealevel)
        self.do_erosion(100, 0.025)

        self.raise_sealevel(np.random.randint(sealevel, sealevel + 20))
        self.clean_coast()
        self.finalize()
        return self.downhill, self.flow, self.slope

    def relax(self):
        newelev = np.zeros_like(self.elevation[:-1])
        for u in range(self.vxs.shape[0]):
            adjs = [v for v in self.adj_vxs[u] if v != -1]
            if len(adjs) < 2: continue
            newelev[u] = np.mean(self.elevation[adjs])
        self.elevation[:-1] = newelev

    def normalize_elevation(self):
        self.elevation -= self.elevation.min()
        if self.elevation.max() > 0:
            self.elevation /= self.elevation.max()
        self.elevation[self.elevation < 0] = 0
        self.elevation = self.elevation ** 0.5