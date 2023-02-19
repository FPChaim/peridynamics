'''LPS 2D model. Also seem as StateBased'''

import numpy as np
import peridynamics as pd
from peridynamics._Models.Base import Base

class LPS_2D(Base):
    '''LPS 2D model. Also seem as StateBased
    
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
        super().__init__('LPS 2D')

        self.name='LPS 2D'
        # Boolean parameters
        self.dilatation=True
        self.dilatHt=False # ?
        self.linearity=False
        self.stiffnessAnal=False
        # Micro modulus

        # Critical relative elongation
        δ=par_omega[0]
        # Damage parameters - no damage dependent Sc
        bc.damage.alpha=0
        bc.damage.beta=0
        bc.damage.gamma=1
        # self.number=4

        # Force vector state
        def T():
            # to be added
            pass
        self.T=T