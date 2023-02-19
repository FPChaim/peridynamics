'''LSJ-T model. Also seem as LSJT'''

import numpy as np
import peridynamics as pd
from peridynamics._Models.Base import Base

class LSJ_T(Base):
    '''LSJ-T model. Also seem as LSJT
    
    Class so that one can initiate the model object with built-in attributes that will be used throughout the script'''

    def __init__(self,par_omega,E,G0,bc) -> None:
        '''- c: micro-modulus | type: np.ndarray
        - dilatation: informs if the model considers dilatation or not |type: boolean
        - dilatHt: ? |type: boolean
        - linearity: informs if the model is linear with the displacement field or not |type: boolean
        - name: model name: "LBB", "LPS 2D", "LSJ-T", "Lipton Free Damage", "PMB" or "PMB DTT" | type: str
        - stiffnessAnal: informs if the model has an analytical way of calculating the stiffness matrix |type: boolean
        - Sc: critical relative elongation | type: float
        - T: interaction force | type: function'''
        super().__init__('LSJ-T')

        self.name='LSJ-T'
        # Boolean parameters
        self.dilatation=True
        self.dilatHt=True
        self.linearity=True
        self.stiffnessAnal=False
        # Micro modulus

        # Critical relative elongation
        δ=par_omega[0]
        # Damage parameters
        bc.damage.alpha=None # ?
        bc.damage.beta=None # ?
        bc.damage.gamma=None # ?
        # self.number=6
        # Force vector state
        def T():
            # to be added
            pass
        self.T=T