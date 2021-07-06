
from itertools import product

import torch
from math import sqrt
from config import *

class Priors:
    def __init__(self,clip = True):
        self.image_size = image_size
        self.strides = strides
        self.feature_maps = feature_maps
        self.clip = clip
    def __call__(self):
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):

                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale
                scale_factor = scale_factors[k] * 0.5 / self.image_size

                priors.append([cx, cy, scale_factor])

        priors = torch.tensor(priors) #(num_priors,4)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
def get_priors():
    return Priors()
if __name__ == "__main__":
    priors = Priors()
    print(priors().size())
