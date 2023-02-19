'''Wrap everything together'''

from peridynamics import BoundaryConditions
from peridynamics._Models.LBB import LBB
from peridynamics._Models.Lipton_Free_Damage import Lipton_Free_Damage
from peridynamics._Models.LPS_2D import LPS_2D
from peridynamics._Models.LSJ_T import LSJ_T
from peridynamics._Models.PMB import PMB
from peridynamics._Models.PMB_DTT import PMB_DTT

def Model(name:str,par_omega:tuple[float|int,int,int],E:float|int,G0:float|int,bc:BoundaryConditions) -> LBB|Lipton_Free_Damage|LPS_2D|LSJ_T|PMB|PMB_DTT:
    '''Returns an instantiated class object of a model with the following attributes:
        - c: micro-modulus | type: np.ndarray
        - dilatation: informs if the model considers dilatation or not |type: boolean
        - dilatHt: ? |type: boolean
        - linearity: informs if the model is linear with the displacement field or not |type: boolean
        - name: model name: "LBB", "LPS 2D", "LSJ-T", "Lipton Free Damage", "PMB" or "PMB DTT" | type: str
        - stiffnessAnal: informs if the model has an analytical way of calculating the stiffness matrix |type: boolean
        - Sc: critical relative elongation | type: float
        - T: interaction force | type: function'''

    if name=='LBB':
        return LBB(par_omega,E,G0,bc)
    elif name=='Lipton Free Damage':
        return Lipton_Free_Damage(par_omega,E,G0,bc)
    elif name=='LPS 2D':
        return LPS_2D(par_omega,E,G0,bc)
    elif name=='LSJ-T':
        return LSJ_T(par_omega,E,G0,bc)
    elif name=='PMB':
        return PMB(par_omega,E,G0,bc)
    elif name=='PMB DTT':
        return PMB_DTT(par_omega,E,G0,bc)
    else:
        raise NotImplementedError(f'Model {name} was not implemented. Available options are: "LBB", "Lipton Free Damage", "LPS 2D", "LSJ-T", "PMB" or "PMB-DTT"')
