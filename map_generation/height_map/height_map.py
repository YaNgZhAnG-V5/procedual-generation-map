import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as spa

from map_generation.utils import *


class HeightMap:
    def __init__(self, grid):
        self.grid = grid
        self.mode = self.grid.mode
        self.erodability = self.grid.erodability
        self.elevation = self.grid.elevation
        self.vxs = self.grid.vxs
        self.dvxs = self.grid.dvxs
        self.adj_vxs = self.grid.adj_vxs
        self.adj_mat = self.grid.adj_mat
        self.edge = self.grid.edge
        self.regions = self.grid.regions
        self.pts = self.grid.pts

        self.downhill, self.flow, self.slope = None, None, None

    def get_heightmap(self):
        pass

    def clean_coast(self, n=3, outwards=True):
        for _ in range(n):
            new_elev = self.elevation[:-1].copy()
            for u in range(self.vxs.shape[0]):
                if self.edge[u] or self.elevation[u] <= 0:
                    continue
                adjs = self.adj_vxs[u]
                adjelevs = self.elevation[adjs]
                if np.sum(adjelevs > 0) == 1:
                    new_elev[u] = np.mean(adjelevs[adjelevs <= 0])
            self.elevation[:-1] = new_elev
            if outwards:
                for u in range(self.vxs.shape[0]):
                    if self.edge[u] or self.elevation[u] > 0:
                        continue
                    adjs = self.adj_vxs[u]
                    adjelevs = self.elevation[adjs]
                    if np.sum(adjelevs <= 0) == 1:
                        new_elev[u] = np.mean(adjelevs[adjelevs > 0])
                self.elevation[:-1] = new_elev

    def raise_sealevel(self, perc=35):
        maxheight = self.elevation.max()
        self.elevation -= np.percentile(self.elevation, perc)
        self.elevation *= maxheight / self.elevation.max()
        self.elevation[-1] = 0

    def do_erosion(self, n, rate=0.01):
        for _ in range(n):
            self.calc_downhill()
            self.calc_flow()
            self.calc_slopes()
            self.erode(rate)
            self.infill()
            self.elevation[-1] = 0

    def finalize(self):
        self.calc_downhill()
        self.infill()
        self.calc_downhill()
        self.calc_flow()
        self.calc_slopes()
        self.calc_elevation_pts()

    def calc_downhill(self):
        n = self.vxs.shape[0]
        dhidxs = np.argmin(self.elevation[self.adj_mat], 1)
        downhill = self.adj_mat[np.arange(n), dhidxs]
        downhill[self.elevation[:-1] <= self.elevation[downhill]] = -1
        downhill[self.edge] = -1
        self.downhill = downhill

    def calc_flow(self):
        n = self.vxs.shape[0]
        rain = np.ones(n) / n
        i = self.downhill[self.downhill != -1]
        j = np.arange(n)[self.downhill != -1]
        dmat = spa.eye(n) - spa.coo_matrix((np.ones_like(i), (i, j)), (n, n)).tocsc()
        self.flow = sla.spsolve(dmat, rain)
        self.flow[self.elevation[:-1] <= 0] = 0

    def calc_slopes(self):
        dist = distance(self.vxs, self.vxs[self.downhill, :])
        self.slope = (self.elevation[:-1] - self.elevation[self.downhill]) / (dist + 1e-9)
        self.slope[self.downhill == -1] = 0

    def calc_elevation_pts(self):
        npts = self.pts.shape[0]
        self.elevation_pts = np.zeros(npts)
        for p in range(npts):
            if self.regions[p]:
                self.elevation_pts[p] = np.mean(self.elevation[self.regions[p]])

    def erode(self, max_step=0.05):
        riverrate = -self.flow ** 0.5 * self.slope  # river erosion
        sloperate = -self.slope ** 2 * self.erodability  # slope smoothing
        rate = 1000 * riverrate + sloperate
        rate[self.elevation[:-1] <= 0] = 0
        self.elevation[:-1] += rate / np.abs(rate).max() * max_step

    def infill(self):
        tries = 0
        while True:
            tries += 1
            sinks = self.get_sinks()
            if np.all(sinks == -1):
                if tries > 1:
                    print(tries, "tries")
                return
            if tries == 1:
                print("Infilling", np.sum(sinks != -1), np.mean(self.vxs[sinks
                                                                         != -1, :], 0), )
            h, u, v = self.find_lowest_sill(sinks)
            sink = sinks[u]
            if self.downhill[v] != -1:
                self.elevation[v] = self.elevation[self.downhill[v]] + 1e-5
            sinkelev = self.elevation[:-1][sinks == sink]
            h = np.where(sinkelev < h, h + 0.001 * (h - sinkelev), sinkelev) + 1e-5
            self.elevation[:-1][sinks == sink] = h
            self.calc_downhill()

    def get_sinks(self):
        sinks = self.downhill.copy()
        water = self.elevation[:-1] <= 0
        sinklist = np.where((sinks == -1) & ~water & ~self.edge)[0]
        sinks[sinklist] = sinklist
        sinks[water] = -1
        while True:
            newsinks = sinks.copy()
            newsinks[~water] = sinks[sinks[~water]]
            newsinks[sinks == -1] = -1
            if np.all(sinks == newsinks): break
            sinks = newsinks
        return sinks

    def find_lowest_sill(self, sinks):
        h = 10000
        edges = np.where((sinks != -1) & np.any((sinks[self.adj_mat] == -1) & self.adj_mat != -1, 1))[0]

        for u in edges:
            adjs = [v for v in self.adj_vxs[u] if v != -1]
            for v in adjs:
                if sinks[v] == -1:
                    newh = max(self.elevation[v], self.elevation[u])
                    if newh < h:
                        h = newh
                        bestuv = u, v
        assert h < 10000
        u, v = bestuv
        return h, u, v

    def rift(self):
        v = np.random.normal(0, 5, (1, 2))
        side = 20 * (distance(self.dvxs, 0.5) ** 2 - 1)
        value = np.random.normal(0, 0.3)
        self.elevation[:-1] += np.arctan(side) * value
