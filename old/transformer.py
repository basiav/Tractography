from typing import TypeAlias, Union
from KernelPCA import TD_Kernel_PCA
from TractDensityComponents import TD_PCA
from nmf import NMFProjections


Projector: TypeAlias = Union[TD_Kernel_PCA, TD_PCA, NMFProjections]

class Transformer:
    def __init__(self, projector: Projector):
        self.projector = projector

    def fit_transform(self, norm, **kwargs):
        return self.projector.fit_transform_maps(norm=norm, **kwargs)
    
    def transform(self, maps):
        return self.projector.transform_maps(maps)