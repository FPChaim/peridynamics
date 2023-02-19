'''LBB model. Also seem as LLPSBB'''

import numpy as np
import peridynamics as pd
from peridynamics._Models.Base import Base

class LBB(Base):
    '''LBB model. Also seem as LLPSBB
    
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
        super().__init__('LBB')
        
        self.name='LBB'
        # Boolean parameters
        self.dilatation=False # True if the model has a dilatation term
        self.dilatHt=False # ?
        self.linearity=True # True if the model is linear with the displacement field
        self.stiffnessAnal=False # True if there is a analyticalStiffnessMatrix implemented
        # Micro modulus
        mm=pd.general_functions.weightedVolume(par_omega)
        self.c=np.array([6*E/mm])
        # Critical relative elongation
        # δ=par_omega[0]
        bc.damage.Sc=G0/4
        # Damage parameters
        bc.damage.alpha=None # ?
        bc.damage.beta=None # ?
        bc.damage.gamma=None # ?
        # self.number=2

        # Force vector state
        def T(mesh,u,ii,family,bc):
            '''Force vector state
            
            Input parameters:
            
            - mesh: mesh generated with Mesh
            - u: degree of freedom displacement vector ? | displacement of the nodes ? | 2N
            - ii: index of the i-th node
            - family: family generated with Family
            - bc: boundary conditions generated with BoundaryConditions
            
            Output parameters:
            
            - f: vector state force between j and i nodes
            - µ: damage factor
            - Also updates model.history.S (maximum stretch for the given bond)'''
            
            # Initiate history
            # here the user should run the initialize_history(mesh,family) method because the mesh and the family are independent from the model
            # self._family=family

            # x_i=x[ii,:] # 2
            # x_j=x[jj,:] # Nx2
            x_i=mesh.points[ii] # (2,)
            x_j=family.xj[ii] # (neigh,2)
            # dofi=dof_vec[ii,:] # 2
            # dofj=dof_vec[jj,:] # Nx2
            dofi=bc.dofi[ii] # (2,)
            try:
                dofj=bc.dofj[ii] # (neigh,2)
            except AttributeError:
                bc.compute_dofj(family)
                dofj=bc.dofj[ii] # (neigh,2)
            u_i=u[dofi] # (2,)
            u_j=u[dofj] # (neigh,2)
            xi=x_j-x_i # \xi | (neigh,2)
            eta=u_j-u_i # \eta | (neigh,2)
            norma=np.linalg.norm(xi,axis=1) # (neigh,)
            S=np.einsum('ij,ij->i',eta,xi)/norma**2 # (neigh,)
            #S=np.dot(eta,xi)/norma**2 # Calculate stretch | (neigh,)
            # keepdimns will make make a Nx1 np.linalg.norm np.array instead of N
            ee=xi/norma.reshape((-1,1)) # Versor | (neigh,2) | Tulio eq 2.5
            
            # Update maximum stretch history
            # try:
            #     # self.history_S[S>self.history_S]=S[S>self.history_S]
            #     self.history_S[ii][S>self.history_S[ii]]=S[S>self.history_S[ii]]
            # except AttributeError:
            #     raise UserWarning('Please run the method initialize_history() to proceed')
            self.update_history_S(ii,S)    
            
            # µ=damageFactor(self.history_S[ii],ii,family,bc) # (neigh,)
            µ=1.

            # Evaluating the force interaction
            f=self.c[0]*(pd.general_functions.influenceFunction(norma,par_omega)*norma*S*µ).reshape((-1,1))*ee # (neigh,) * (neigh,2) --> (neigh,2)
            
            return f,µ
        self.T=T
        
        def damageFactor(x,ii,family,bc):
            '''Input parameters:
            
            - x: stretch, js integral, jtheta integral ...
            - ii: index of node i
            - family: family generated with Family
            - bc: boundary conditions generated with BoundaryConditions

            Output parameters:

            - µ: damage factor for every neighbot of ii'''
            
            # Preallocate the damage factor
            if bc.damage.damage_on==True:
                µ=(x<bc.damage.Sc)*1.
            else:
                µ=np.ones_like(x)

            # Deal with initial broken bonds
            # brokenBonds=bc.damage.brokenBonds[ii,neighIndex]
            # brokenBonds=bc.damage.brokenBonds[ii,:len(family.j[ii])]
            brokenBonds=bc.damage.brokenBonds[ii]
            if np.any(brokenBonds)==True:
                #µ[brokenBonds]=np.zeros((np.sum(brokenBonds),µ.shape[1]))
                µ[brokenBonds]=0.
            
            if bc.noFail.size>0:
                #µ[noFail,:]=np.ones(µ[noFail,:].shape)
                try:
                    µ[bc.noFail_ij[ii]]=1.
                except AttributeError:
                    bc.compute_noFail_ij(family)
                    µ[bc.noFail_ij[ii]]=1.
                        
            return µ # (neigh,)

        def strainEnergyDensity(mesh,u,family,ii,bc,par_omega) -> float:
            '''Input parametesr:
            
            - mesh: mesh generated with Mesh 
            - u: displacements | 2N
            - family: family generated with Family
            - ii: node index
            - bc: boundary conditions generated with BoundaryCondition
            - par_omega ([δ,ωδ,γ]): influence function parameters
                - horizon (δ): peridynamic horizon
                - omega (ωδ): normalized function type
                    - 1: Exponential
                    - 2: Constant
                    - 3: Conical
                    - 4: Cubic polynomial
                    - 5: P5
                    - 6: P7
                    - 7: Singular
                - gamma (γ): integer
            
            Output parameters:
            
            -W: strain energy density of a bond'''
            
            #familySet=family[family>-1] # neigh
            #familySet=family.compressed() # neigh
            # jj=family.j[ii] # neigh
            # partialAreas=family.PAj[ii]
            # surfaceCorrection=family.SCj[ii]
            # dofi=np.array([idb[2*ii],idb[2*ii+1]]) # 2
            # idb=bc.idb[ii]
            dofi=bc.dofi[ii]

            # neigh_ind=np.arange(0,len(jj)) # neigh

            # xi=x[jj,:]-x[ii,:] # neigh x 2
            xi=family.xj[ii]-mesh.points[ii] # neigh x 2
            # dofj=np.array([idb[2*jj],idb[2*jj+1]]).reshape(-1,2) # neigh x 2
            # dofj=column_stack((idb[2*jj],idb[2*jj+1])) # neigh x 2
            try:
                dofj=bc.dofj[ii] # (j,2)
            except AttributeError:
                bc.compute_dofj(family)
                dofj=bc.dofj[ii] # (j,2)
            
            eta=u[dofj]-u[dofi]
            norma=np.linalg.norm(xi,axis=1)
            # s=(np.linalg.norm(xi+eta,axis=1)-norma)/norma
            s=np.einsum('ij,ij->i',eta,xi)/norma**2 # (neigh,)

            # noFail=logical_or(bc.damage.noFail[ii],bc.damage.noFail[jj])
            # µ=damageFactor(self.history_S[neigh_ind],ii,neigh_ind,bc) # NoFail not required
            # µ=damageFactor(self.history_S[neigh_ind],ii,family,bc) # NoFail not required
            µ=damageFactor(self.history.S[ii],ii,family,bc) # NoFail not required
            p=µ*s**2/2
            w=1/2*self.c[0]*pd.general_functions.influenceFunction(norma,par_omega)*norma**2*p*µ
            #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
            #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
            W=np.sum(w*family.PAj[ii]*family.SCj[ii])
                        
            return W
        self.strainEnergyDensity=strainEnergyDensity