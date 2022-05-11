# coding: utf-8

from language import Language
from civilization import CitiesPlacement
from plotter import Plotter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.spatial as spl
import scipy.sparse as spa
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as sla
import heapq
import noise
import itertools
from utils import *

plt.rcParams['font.family'] = "Palatino Linotype"
plt.rcParams['font.size'] = 10


class MapGrid(object):
    def __init__(self, mode='shore', n=16384):
        self.mode = mode
        self.lang = Language()
        self.build_grid(n)

        if '/' in mode:
            self.mixed_heightmap()
            mode = mode.split("/")[0]
        else:
            self.single_heightmap(mode)
        self.finalize()
        self.riverperc = riverpercs[mode] * np.mean(self.elevation > 0)

        cities_placer = CitiesPlacement(self.flow, self.elevation, self.vxs, self.adj_vxs, self.edge_weight)
        self.cities = cities_placer.place_cities(np.random.randint(*city_counts[mode]))
        self.territories = cities_placer.grow_territory(np.random.randint(*terr_counts[mode]))
        self.city_names, self.region_names = self.lang.name_places(self.cities, self.territories)

        self.path_cache = {}
        self.fill_path_cache(self.big_cities)

    @property
    def big_cities(self):
        return [c for c in self.cities if self.territories[c] == c]

    def save(self, filename):
        with gzip.open(filename, "w") as f:
            f.write(pickle.dumps(self))

    def build_grid(self, n):
        self.pts = np.random.random((n, 2))
        self.improve_pts()
        self.vor = spl.Voronoi(self.pts)
        self.regions = [self.vor.regions[i] for i in self.vor.point_region]
        self.vxs = self.vor.vertices
        self.build_adjs()
        self.improve_vxs()
        self.calc_edges()
        self.distort_vxs()
        self.elevation = np.zeros(self.vxs.shape[0] + 1)
        self.erodability = np.ones(self.vxs.shape[0])

    def do_erosion(self, n, rate=0.01):
        for _ in range(n):
            self.calc_downhill()
            self.calc_flow()
            self.calc_slopes()
            self.erode(rate)
            self.infill()
            self.elevation[-1] = 0

    def raise_sealevel(self, perc=35):
        maxheight = self.elevation.max()
        self.elevation -= np.percentile(self.elevation, perc)
        self.elevation *= maxheight / self.elevation.max()
        self.elevation[-1] = 0

    def finalize(self):
        self.calc_downhill()
        self.infill()
        self.calc_downhill()
        self.calc_flow()
        self.calc_slopes()
        self.calc_elevation_pts()

    def rift(self):
        v = np.random.normal(0, 5, (1, 2))
        side = 20 * (distance(self.dvxs, 0.5) ** 2 - 1)
        value = np.random.normal(0, 0.3)
        self.elevation[:-1] += np.arctan(side) * value

    def improve_pts(self, n=2):
        print("Improving points")
        for _ in range(n):
            vor = spl.Voronoi(self.pts)
            newpts = []
            for idx in range(len(vor.points)):
                pt = vor.points[idx, :]
                region = vor.regions[vor.point_region[idx]]
                if -1 in region:
                    newpts.append(pt)
                else:
                    vxs = np.asarray([vor.vertices[i, :] for i in region])
                    vxs[vxs < 0] = 0
                    vxs[vxs > 1] = 1
                    newpt = np.mean(vxs, 0)
                    newpts.append(newpt)
            self.pts = np.asarray(newpts)

    def improve_vxs(self):
        print("Improving vertices")
        n = self.vxs.shape[0]
        for v in range(n):
            self.vxs[v, :] = np.mean(self.pts[self.vx_regions[v]], 0)

    def build_adjs(self):
        print("Building adjacencies")
        self.adj_pts = defaultdict(list)
        self.adj_vxs = defaultdict(list)
        for p1, p2 in self.vor.ridge_points:
            self.adj_pts[p1].append(p2)
            self.adj_pts[p2].append(p1)
        for v1, v2 in self.vor.ridge_vertices:
            self.adj_vxs[v1].append(v2)
            self.adj_vxs[v2].append(v1)
        self.vx_regions = defaultdict(list)
        for p in range(self.pts.shape[0]):
            for v in self.regions[p]:
                if v != -1:
                    self.vx_regions[v].append(p)
        self.tris = defaultdict(list)
        for p in range(self.pts.shape[0]):
            for v in self.regions[p]:
                self.tris[v].append(p)
        self.adj_mat = np.zeros((self.vxs.shape[0], 3), np.int32) - 1
        for k, v in self.adj_vxs.items():
            if k != -1:
                self.adj_mat[k, :] = v

    def calc_edges(self):
        n = self.vxs.shape[0]
        self.edge = np.zeros(n, bool)
        for u in range(n):
            adjs = self.adj_vxs[u]
            if -1 in adjs:  # or \
                #                     np.all(self.vxs[adjs,0] > self.vxs[u,0]) or \
                #                     np.all(self.vxs[adjs,0] < self.vxs[u,0]) or \
                #                     np.all(self.vxs[adjs,1] > self.vxs[u,1]) or \
                #                     np.all(self.vxs[adjs,1] < self.vxs[u,1]):
                self.edge[u] = True

    def perlin(self, base=None):
        if base is None:
            base = np.random.randint(1000)
        return np.array([noise.pnoise2(x, y, lacunarity=1.7, octaves=3,
                                       base=base) for x, y in self.vxs])

    def distort_vxs(self):
        self.dvxs = self.vxs.copy()
        self.dvxs[:, 0] += self.perlin()
        self.dvxs[:, 1] += self.perlin()

    def single_heightmap(self, mode):
        modefunc = getattr(self, mode + "_heightmap")
        modefunc()
        return self.elevation[:-1].copy()

    def mixed_heightmap(self):
        mode1, mode2 = self.mode.split("/")
        hm1 = self.single_heightmap(mode1)
        print("HM1:", mode1, hm1.max(), hm1.min(), hm1.mean())
        hm2 = self.single_heightmap(mode2)
        print("HM2:", mode2, hm2.max(), hm2.min(), hm2.mean())
        mix = 20 * (self.dvxs[:, 0] - self.dvxs[:, 1])
        mixing = 1 / (1 + np.exp(-mix))
        print("MIX:", mixing.max(), mixing.min(), mixing.mean())
        self.elevation[:-1] = mixing * hm2 + (1 - mixing) * hm1
        self.clean_coast()

    def shore_heightmap(self):
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

    def island_heightmap(self):
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

    def mountain_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = 0.5 * (self.vxs + self.dvxs)
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = 50 - 10 * np.abs(x)
        mountains = np.random.random((50, 2))
        for m in mountains:
            self.elevation[:-1] += np.exp(-distance(self.vxs, m) ** 2 / (2 * 0.05 ** 2)) ** 2
        self.erodability[:] = np.exp(50 - 10 * self.elevation[:-1])
        for _ in range(5):
            self.rift()
        self.do_erosion(250, 0.02)
        self.elevation *= 0.5
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.5

    def desert_heightmap(self):
        theta = np.random.random() * 2 * np.pi
        dvxs = self.dvxs
        x = (dvxs[:, 0] - .5) * np.cos(theta) + (dvxs[:, 1] - .5) * np.sin(theta)
        y = (dvxs[:, 0] - .5) * -np.sin(theta) + (dvxs[:, 1] - .5) * np.cos(theta)
        self.elevation[:-1] = x + 20 + 2 * self.perlin()
        self.erodability[:] = 1
        self.do_erosion(50, 0.005)
        self.elevation[:-1] -= self.elevation[:-1].min() - 0.1

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

    def calc_elevation_pts(self):
        npts = self.pts.shape[0]
        self.elevation_pts = np.zeros(npts)
        for p in range(npts):
            if self.regions[p]:
                self.elevation_pts[p] = np.mean(self.elevation[self.regions[p]])

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

    def erode(self, max_step=0.05):
        riverrate = -self.flow ** 0.5 * self.slope  # river erosion
        sloperate = -self.slope ** 2 * self.erodability  # slope smoothing
        rate = 1000 * riverrate + sloperate
        rate[self.elevation[:-1] <= 0] = 0
        self.elevation[:-1] += rate / np.abs(rate).max() * max_step

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
        edges = np.where((sinks != -1) & \
                         np.any(
                             (sinks[self.adj_mat] == -1) &
                             self.adj_mat != -1, 1))[0]

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
                                                                   != -1, :], 0),)
            h, u, v = self.find_lowest_sill(sinks)
            sink = sinks[u]
            if self.downhill[v] != -1:
                self.elevation[v] = self.elevation[self.downhill[v]] + 1e-5
            sinkelev = self.elevation[:-1][sinks == sink]
            h = np.where(sinkelev < h, h + 0.001 * (h - sinkelev), sinkelev) + 1e-5
            self.elevation[:-1][sinks == sink] = h
            self.calc_downhill()

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

    def edge_weight(self, u, v, territory=False):
        horiz = distance(self.vxs[u, :], self.vxs[v, :])
        vert = self.elevation[v] - self.elevation[u]
        if vert < 0:
            vert /= 10
        difficulty = 1 + (vert / horiz) ** 2
        if territory:
            difficulty += 100 * self.flow[u] ** 0.5
        else:
            if self.downhill[u] == v or self.downhill[v] == u:
                difficulty *= 0.9
        if self.elevation[u] <= 0:
            difficulty = 1
        if (self.elevation[u] <= 0) != (self.elevation[v] <= 0):
            difficulty = 2000 if territory else 200
        return horiz * difficulty

    def fill_path_cache(self, cities):
        cities = list(cities)
        n = self.vxs.shape[0]
        g = spa.dok_matrix((n + 2, n + 2))
        edge = self.extend_area(self.edge, 5)
        for u in range(n):
            for v in self.adj_vxs[u]:
                if not (edge[u] and edge[v]):
                    g[u, v] = self.edge_weight(u, v)
            if edge[u]:
                d = self.vxs[u, 0] - self.vxs[u, 1]
                if d < -0.5:
                    g[n, u] = 1e-12
                if d > 0.5:
                    g[u, n + 1] = 1e-12
        g = g.tocsc()
        tocities = cities + [n + 1]
        fromcities = cities + [n]
        dists, preds = csg.dijkstra(g,
                                    indices=fromcities,
                                    return_predecessors=True)

        for b in tocities:
            for i, a in enumerate(fromcities):
                if a == b: continue
                p = [b]
                while p[0] != a:
                    p.insert(0, preds[i, p[0]])
                p = [x for x in p if x < n]
                d = dists[i, b]
                self.path_cache['topleft' if a == n else a,
                                'bottomright' if b == n + 1 else b] = p, d

    def shortest_path(self, start, end, ):
        try:
            return self.path_cache[start, end]
        except KeyError:
            print("WARNING: Uncached path search", start, end)
            pass
        best_dir = {}
        q = []
        flipped = False
        if isinstance(end, str):
            flipped = True
            start, end = end, start
        heapq.heappush(q, (0, 0, end, -1))
        while start not in best_dir:
            _, dist, u, v = heapq.heappop(q)
            if u in best_dir:
                continue
            best_dir[u] = v
            if self.territories[u] == u:
                path = [u]
                while path[-1] != end:
                    path.append(best_dir[path[-1]])
                if flipped:
                    self.path_cache[end, u] = path[::-1], dist
                    print("CACHE", len(self.path_cache))
                else:
                    self.path_cache[u, end] = path, dist
                    print("CACHE", len(self.path_cache))
            length = dist
            if isinstance(start, str) and self.edge[u]:
                if (start == "topleft" and
                    self.vxs[u, 0] - self.vxs[u, 1] < -0.5) or \
                        (start == "bottomright" and
                         self.vxs[u, 0] - self.vxs[u, 1] > 0.5):
                    start = u
                    break
            for w in self.adj_vxs[u]:
                if w == -1 or w in best_dir or (self.edge[u] and self.edge[w]):
                    continue
                est = distance(self.vxs[end, :], self.vxs[w, :]) * 0.85
                d = dist + self.edge_weight(w, u)
                heapq.heappush(q, (d + est, d, w, u))
        path = [start]
        while path[-1] != end:
            path.append(best_dir[path[-1]])
        if flipped:
            self.path_cache[end, start] = path[::-1], length
            print("CACHE", len(self.path_cache))
            return path[::-1], length
        else:
            self.path_cache[start, end] = path, length
            print("CACHE", len(self.path_cache))
            return path, length

    def extend_area(self, area, n):
        for _ in range(10):
            adj = self.adj_mat[area, :]
            area[adj[adj != -1]] = True
        return area

    def ordered_cities(self):
        cities = self.big_cities
        dists = {}
        for c in cities:
            for d in cities:
                if c == d:
                    continue
                dists[c, d] = self.shortest_path(c, d)[1]
            dists["topleft", c] = self.shortest_path("topleft", c)[1]
            dists[c, "bottomright"] = self.shortest_path(c, "bottomright")[1]

        def totallength(seq):
            seq = ["topleft"] + list(seq) + ["bottomright"]
            return sum(dists[c, d] for c, d in zip(seq[:-1], seq[1:]))

        clist = min(itertools.permutations(cities),
                    key=totallength)
        return clist


if __name__ == '__main__':
    for i in range(10):
        for mode in ["shore", "island", "mountain", "desert"]:
            plt.close('all')
            while True:
                try:
                    m = MapGrid(mode=mode)
                    filename = "tests/%s-%02d.png" % (m.mode, i)
                    plotter = Plotter(m)
                    plotter.plot(filename)
                    break
                except AssertionError:
                    print("Failed assertion, retrying")
