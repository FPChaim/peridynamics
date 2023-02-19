'''Functions related to the boundary conditions'''

######################################################################################################################################################
# BOUNDARY CONDITIONS STUFF ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
######################################################################################################################################################

class Damage:
    '''Object with built-in attributes that will be used throughout the script'''
    def __init__(self) -> None:
        '''- alpha: damage parameter
        - beta: damage parameter
        - brokenBonds (only if the methdod initialize_brokenBonds(family) was used): brokenBonds vector (initialization only) | ii: node, jj: neighbors, kk: crack segment i to i+1
        - crackIn: Nx2 np.array of initial crack segments, where N≥2
        - damage_dependent_Sc: True if Sc is damage dependent
        - damage_on: True if applying damage to the model
        - gamma: damage parameter
        - phi: damage index (actually not touched here. Calculated in the solver)
        - Sc: critical relative elongation
        - thetaC: dilatation parameter ?'''
        self.alpha=None
        self.beta=None
        self.brokenBonds=None
        # self.checkCrack=None
        self.crackIn=None
        self.damage_dependent_Sc=None
        self.damage_on=None
        self.gamma=None
        # self.noFail=None
        self.phi=None
        self.Sc=None
        self.thetaC=None
    