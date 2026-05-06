from dataset import Dataset


class Datasets:
    UCSF = Dataset("ucsf", "path/1", maps_ucsf, data_ucsf)
    UPENN = Dataset("upenn", "path/2", maps_upenn, data_upenn)
    RHUH = Dataset("rhuh", "path/3", maps_rhuh, data_rhuh)
    TCGA = Dataset("tcga", "path/4", maps_tcga, data_tcga)


import itertools


possibilities = [Datasets.UCSF, Datasets.UPENN, Datasets.RHUH, Datasets.TCGA]
pairs = list(itertools.permutations(possibilities, 2))