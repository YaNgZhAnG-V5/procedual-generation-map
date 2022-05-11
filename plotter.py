import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils import *


class Plotter:
    def __init__(self, map_grid):
        self.map_grid = map_grid

    def plot(self, filename, rivers=True, cmap=mpl.cm.Greys, **kwargs):
        print("Plotting")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0, 0, 1, 1])

        elev = np.where(self.map_grid.elevation > 0, 0.1, 0)
        good = ~self.map_grid.extend_area(self.map_grid.edge, 10)

        goodidxs = [i for i in range(self.map_grid.vxs.shape[0]) if good[i]]
        tris = [self.map_grid.tris[i] for i in goodidxs]
        elevs = elev[goodidxs]

        slopelines = []
        r = 0.25 * self.map_grid.vxs.shape[0] ** -0.5
        for i in goodidxs:
            if self.map_grid.elevation[i] <= 0: continue
            t = self.map_grid.tris[i]
            s, s2 = trislope(self.map_grid.pts[t, :], self.map_grid.elevation_pts[t])
            s /= 10
            if abs(s) < 0.1 + 0.3 * np.random.random():
                continue
            x, y = self.map_grid.vxs[i, :]
            l = r * (1 + np.random.random()) \
                * (1 - 0.2 * np.arctan(s) ** 2) \
                * np.exp(s2 / 100)
            if abs(l * s) > 2 * r:
                n = int(abs(l * s / r))
                l /= n
                uv = np.random.normal(0, r / 2, (min(n, 4), 2))
                for u, v in uv:
                    slopelines.append([(x + u - l, y + v - l * s), (x + u + l,
                                                                    y + v + l * s)])
            else:
                slopelines.append([(x - l, y - l * s), (x + l, y + l * s)])

        slopecol = mpl.collections.LineCollection(slopelines)
        slopecol.set_zorder(1)
        slopecol.set_color('black')
        slopecol.set_linewidth(0.3)
        ax.add_collection(slopecol)

        #         land = np.where(elevs > 0)[0]
        #         landpatches = [mpl.patches.Polygon(self.pts[tris[i],:], closed=True) for i in land]
        #         landpatchcol = mpl.collections.PatchCollection(landpatches)
        #         landpatchcol.set_color(colors[land,:])
        #         landpatchcol.set_zorder(0)
        #         ax.gca().add_collection(landpatchcol)

        sea = np.where(elevs <= 0)[0]
        seapatches = [mpl.patches.Polygon(self.map_grid.pts[tris[i], :], closed=True) for i in sea]
        seapatchcol = mpl.collections.PatchCollection(seapatches)
        seapatchcol.set_color('white')
        seapatchcol.set_zorder(10)
        ax.add_collection(seapatchcol)

        if rivers:
            land = good & (self.map_grid.elevation[:-1] > 0) & (self.map_grid.downhill != -1) & \
                   (self.map_grid.flow > np.percentile(self.map_grid.flow, 100 - self.map_grid.riverperc))
            rivers = relaxpts(self.map_grid.vxs, [(u, self.map_grid.downhill[u]) for u in range(self.map_grid.vxs.shape[0]) if land[u]])
            print(len(rivers), sum(land))
            rivers = mergelines(rivers)
            rivercol = mpl.collections.PathCollection(rivers)
            rivercol.set_edgecolor('black')
            rivercol.set_linewidth(1)
            rivercol.set_facecolor('none')
            rivercol.set_zorder(9)
            ax.add_collection(rivercol)

        bigcities = self.map_grid.big_cities
        smallcities = [c for c in self.map_grid.cities if c not in bigcities]
        ax.scatter(self.map_grid.vxs[bigcities, 0], self.map_grid.vxs[bigcities, 1],
                   c='white', s=100, zorder=15, edgecolor='black', linewidth=2)
        ax.scatter(self.map_grid.vxs[smallcities, 0], self.map_grid.vxs[smallcities, 1],
                   c='black', s=30, zorder=15, edgecolor='none')

        labelbox = dict(
            boxstyle='round,pad=0.1',
            fc='white',
            ec='none'
        )
        for city in self.map_grid.cities:
            ax.annotate(xy=self.map_grid.vxs[city, :], text=self.map_grid.city_names[city],
                        xytext=(0, 12 if city in bigcities else 8),
                        ha='center', va='center',
                        textcoords='offset points',
                        bbox=labelbox,
                        size='small' if city in bigcities else 'x-small',
                        zorder=20.5 if city in bigcities else 20)
        reglabels = []
        for terr in sorted(np.unique(self.map_grid.territories),
                           key=lambda t: np.sum(
                               (self.map_grid.territories == t) & (self.map_grid.elevation[:-1] > 0))):
            name = self.map_grid.region_names[terr]
            w = 0.06 + 0.015 * len(name)
            region = (self.map_grid.territories == terr)
            landregion = region & (self.map_grid.elevation[:-1] > 0)
            scores = np.zeros(self.map_grid.vxs.shape[0])
            center = np.mean(self.map_grid.vxs[region, :], 0)
            landcenter = np.mean(self.map_grid.vxs[landregion, :], 0)
            landradius = np.mean(landregion) ** 0.5
            scores = -5000 * distance(self.map_grid.vxs, landcenter)
            scores -= 1000 * distance(self.map_grid.vxs, center)
            scores[~region] -= 3000
            for city in self.map_grid.cities:
                dists = self.map_grid.vxs - self.map_grid.vxs[city, :] - np.array([[0, 0.02]])
                exclude = (np.abs(dists[:, 0]) < w) & (np.abs(dists[:, 1]) < 0.05)
                scores[exclude] -= 4000 if city in bigcities else 500
            for rl in reglabels:
                dists = self.map_grid.vxs - rl
                exclude = (np.abs(dists[:, 0]) < 0.15 + w) & (np.abs(dists[:, 1]) < 0.1)
                scores[exclude] -= 5000

            scores[self.map_grid.elevation[:-1] <= 0] -= 500
            scores[self.map_grid.vxs[:, 0] > 1.06 - w] -= 50000
            scores[self.map_grid.vxs[:, 0] < w - 0.06] -= 50000
            scores[self.map_grid.vxs[:, 1] > 0.97] -= 50000
            scores[self.map_grid.vxs[:, 1] < 0.03] -= 50000
            assert scores.max() > -50000
            xy = self.map_grid.vxs[np.argmax(scores), :]
            # ax.axvspan(xy[0] - w, xy[0] + w, xy[1] - 0.07, xy[1] + 0.03,
            # facecolor='none', edgecolor='red', zorder=19)
            print("Labelling %s at %.1f" % (name, scores.max()))
            reglabels.append(xy)
            ax.annotate(xy=xy, text=name,
                        ha='center', va='center',
                        bbox=labelbox,
                        size='large',
                        zorder=21
                        )

        borders = []
        borderadj = defaultdict(list)
        coasts = []
        for rv, rp in zip(self.map_grid.vor.ridge_vertices, self.map_grid.vor.ridge_points):
            if -1 in rv or -1 in rp:
                continue
            v1, v2 = rv
            p1, p2 = rp
            if not (good[v1] and good[v2]):
                continue
            if self.map_grid.territories[v1] != self.map_grid.territories[v2] \
                    and self.map_grid.elevation[v1] > 0 and self.map_grid.elevation[v2] > 0:
                borders.append((p1, p2))
            if (self.map_grid.elevation[v1] > 0 and self.map_grid.elevation[v2] <= 0) or (
                    self.map_grid.elevation[v2] > 0 and self.map_grid.elevation[v1] <= 0):
                coasts.append(self.map_grid.pts[rp, :])

        borders = mergelines(relaxpts(self.map_grid.pts, borders))
        print("Borders:", len(borders))
        bordercol = mpl.collections.PathCollection(borders)
        bordercol.set_facecolor('none')
        bordercol.set_edgecolor('black')
        bordercol.set_linestyle(':')
        bordercol.set_linewidth(3)
        bordercol.set_zorder(11)
        ax.add_collection(bordercol)

        coastcol = mpl.collections.PathCollection(mergelines(coasts))
        coastcol.set_facecolor('none')
        coastcol.set_edgecolor('black')
        coastcol.set_zorder(12)
        coastcol.set_linewidth(1.5)
        ax.add_collection(coastcol)

        clist = self.map_grid.ordered_cities()
        clist = ["topleft"] + list(clist) + ["bottomright"]
        for c1, c2 in zip(clist[:-1], clist[1:]):
            path, _ = self.map_grid.shortest_path(c1, c2)
            path = mergelines(relaxpts(self.map_grid.vxs, zip(path[:-1], path[1:])))
            pathcol = mpl.collections.PathCollection(path)
            pathcol.set_facecolor('none')
            pathcol.set_edgecolor('black')
            pathcol.set_linestyle('--')
            pathcol.set_linewidth(2.5)
            pathcol.set_zorder(14)
            ax.add_collection(pathcol)
            # plt.plot(self.vxs[path, 0], self.vxs[path, 1], c='red',
            # zorder=10000, linewidth=2, alpha=0.5)

        ax.axis('image')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        # plt.xticks(np.arange(0, 21) * .05)
        # plt.yticks(np.arange(0, 21) * .05)
        # plt.grid(True)
        ax.axis('off')
        plt.savefig(filename, **kwargs)
        plt.close()
