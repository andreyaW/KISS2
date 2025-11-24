from dataclasses import dataclass,field
from sensed_objects.kooN_vote_sensed_comp import KooNVoteSensedComp

import numpy as np
import pandas as pd

@dataclass
class MajorityVoteSensedComp(KooNVoteSensedComp):
    sensor_fusion_method: str = 'majority_vote'
    k: int = field(init=False)  # ovveride the k input requirement (k is set in __post_init__)
    
    def __post_init__(self):
        super().__post_init__()
        self.k = int(self.N // 2 +1) 

    #-------------------------------------------------------------------------
    # OPTIONAL: REPR
    # -------------------------------------------------------------------------
    def __repr__(self):
        return f"<MajorityVoteSensedComp: {self.comp.name}, method={self.sensor_fusion_method}>"
