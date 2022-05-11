import numpy as np


class Language:
    def __init__(self):
        self.cities = ['City' + str(i) for i in range(50)]
        self.regions = ['Region' + str(i) for i in range(50)]

    def name(self, types):
        if types == 'city':
            return self.cities.pop(0)
        else:
            return self.regions.pop(0)

    def name_places(self, cities, territories):
        city_names = {}
        region_names = {}
        for city in cities:
            city_names[city] = self.name("city")
        for region in np.unique(territories):
            region_names[region] = self.name("region")
        return city_names, region_names
