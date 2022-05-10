class LANGUAGE:
    def __init__(self):
        self.cities = ['City' + str(i) for i in range(50)]
        self.regions = ['Region' + str(i) for i in range(50)]

    def name(self, types):
        if types == 'city':
            return self.cities.pop(0)
        else:
            return self.regions.pop(0)


def get_language():
    return LANGUAGE()
