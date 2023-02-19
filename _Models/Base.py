'''Peridynamic model base'''

import numpy as np
import peridynamics as pd

class Base:
    '''Peridynamic model base'''

    def __init__(self,name) -> None:
        '''init'''
        self.name=name

    def initialize_history(self,family) -> None:
        '''Initiate the history dependency variable(s)

        Input parameters:

        - family: family generated with Family

        Output parameters:

        - Initialize self.history.S and self.history.theta (when needed) with zeros'''
                
        self.history=pd.general_functions.History()
        
        if self.name=='PMB':
            self.history.S=[]
            for j in family.j:
                self.history.S.append(np.zeros_like(j,dtype=float)) # S_max
            self.history.S=np.array(self.history.S,dtype=object)
        elif self.name=='LBB':
            self.history.S=[]
            for j in family.j:
                self.history.S.append(np.zeros_like(j,dtype=float)) # S_max
            self.history.S=np.array(self.history.S,dtype=object)
        elif self.name=='Lipton Free Damage':
            # self.history_S=np.zeros((len(mesh.points),family.maxNeigh),dtype=float) # js integral
            self.history.S=[]
            for j in family.j:
                self.history.S.append(np.zeros_like(j,dtype=float)) # js integral
            self.history.S=np.array(self.history.S,dtype=object)
            self.history.theta=np.zeros((len(family.j)),dtype=float) # jtheta-y integral
        elif self.name=='LSJ-T':
            # self.history_S=np.zeros((len(mesh.points),family.maxNeigh),dtype=float) # js integral
            self.history.S=[]
            for j in family.j:
                self.history.S.append(np.zeros_like(j,dtype=float)) # js integral
            self.history.S=np.array(self.history.S,dtype=object)
            self.history.theta=np.zeros((len(family.j)),dtype=float) # jtheta-y integral
        elif self.name=='LPS 2D':
            # self.history_S=np.zeros((len(mesh.points),family.maxNeigh),dtype=float) # js integral
            self.history.S=[]
            for j in family.j:
                self.history.S.append(np.zeros_like(j,dtype=float)) # js integral
            self.history.S=np.array(self.history.S,dtype=object)
            self.history.theta=np.zeros((len(family.j)),dtype=float) # jtheta-y integral
        else:
            raise NameError('Available model names are "PMB", "Lipton Free Damage", "LSJ-T" and "LPS 2D".')

    def update_history_S(self,i,S) -> None:
        '''Input parameters:
        
        - i: node index
        - S: stretch
        
        Output parameters:
        
        - Updates self.history.S (maximum stretch for the given bond)'''

        # try:
        #     self.history.S[i][S>self.history.S[i]]=S[S>self.history.S[i]]
        # except AttributeError:
        #     self.initialize_history(self._family)
        #     self.history.S[i][S>self.history.S[i]]=S[S>self.history.S[i]]

        self.history.S[i][S>self.history.S[i]]=S[S>self.history.S[i]]

    def update_history_theta(self,i,theta) -> None:
        '''Input parameters:
        
        - i: node index
        - theta: 
        
        Output parameters:
        
        - Updates self.history.theta'''

        # try:
        #     self.history.theta[i][theta>self.history.theta[i]]=theta[theta>self.history.theta[i]]
        # except AttributeError:
        #     self.initialize_history(self._family)
        #     self.history.theta[i][theta>self.history.theta[i]]=theta[theta>self.history.theta[i]]

        self.history.theta[i][theta>self.history.theta[i]]=theta[theta>self.history.theta[i]]