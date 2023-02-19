'''PMB model'''

import numpy as np
import peridynamics as pd
from peridynamics._Models.Base import Base

class PMB(Base):
    '''PMB model
    
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
        super().__init__('PMB')

        self.name='PMB'
        # Boolean parameters
        self.dilatation=False
        self.dilatHt=False # ?
        self.linearity=False
        self.stiffnessAnal=False # Should be True, but it's not converging
        # Micro modulus
        mm=pd.general_functions.weightedVolume(par_omega)
        self.c=np.array([6*E/mm])
        # Critical relative elongation
        δ=par_omega[0]
        if par_omega[1]==3 and par_omega[2]==1:
            bc.damage.Sc=(5*np.pi*G0/(9*E*δ))**0.5
        elif par_omega[1]==1 and par_omega[2]==0:
            bc.damage.Sc=((1+1/3)*np.pi*G0*3/(8*E*δ*0.66467))**0.5
        elif par_omega[1]==1 and par_omega[2]==1:
            bc.damage.Sc=(G0*(1/3+1)*np.pi**(3/2)/8/E*(3/δ))
        else:
            raise Warning('Critical bond not defined')
        # Damage parameters
        if bc.damage.damage_dependent_Sc==True:
            bc.damage.alpha=0.2
            bc.damage.beta=0.2
            bc.damage.gamma=1.4
        else:
            bc.damage.alpha=0
            bc.damage.beta=0
            bc.damage.gamma=1
        # self.number=5

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
            self._family=family

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
            S=(np.linalg.norm(eta+xi,axis=1)-norma)/norma # Calculate stretch | (neigh,)
            # keepdimns will make make a Nx1 np.linalg.norm np.array instead of N
            ee=(eta+xi)/np.linalg.norm(eta+xi,axis=1,keepdims=True) # Versor | (neigh,2) | Tulio eq 2.5
            
            # Update maximum stretch history
            # try:
            #     # self.history_S[S>self.history_S]=S[S>self.history_S]
            #     self.history_S[ii][S>self.history_S[ii]]=S[S>self.history_S[ii]]
            # except AttributeError:
            #     raise UserWarning('Please run the method initialize_history() to proceed')
            self.update_history_S(ii,S)    
            
            µ=damageFactor(self.history.S[ii],ii,family,bc) # (neigh,)

            # Defining fscalar
            if bc.damage.damage_on==True:
                # Damage dependent crack
                if bc.damage.phi[ii]>bc.damage.alpha and ~np.isclose(bc.damage.phi[ii],1):
                    Sc=bc.damage.Sc*np.min([bc.damage.gamma,1+bc.damage.beta*(bc.damage.phi[ii]-bc.damage.alpha/(1-bc.damage.phi[ii]))])
                else:
                    Sc=bc.damage.Sc

                ff=(S<Sc)*S # N
                ff[bc.noFail_ij[ii]]=S[bc.noFail_ij[ii]]
            else:
                ff=S

            # Evaluating the force interaction
            # Influence function times norma because the omega_d used is related to the original influence function 
            # by omega_d = omega*|\xi|
            f=self.c[0]*(pd.general_functions.influenceFunction(norma,par_omega)*norma*ff*µ).reshape((-1,1))*ee # (neigh,) * (neigh,2) --> (neigh,2)
            
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
            
            # ε=float_info.epsilon
            
            if bc.damage.damage_on==True:
                # Damage dependent crack
                if bc.damage.phi[ii]>bc.damage.alpha and ~np.isclose(bc.damage.phi[ii],1):
                    Sc=bc.damage.Sc*np.min([bc.damage.gamma,\
                        1+bc.damage.beta*(bc.damage.phi[ii]-bc.damage.alpha)/(1-bc.damage.phi[ii])])
                else:
                    Sc=bc.damage.Sc

                µ=(x<Sc)*1.

            else:
                # Preallocate the damage factor
                µ=np.ones((len(family.j[ii])),dtype=float)

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

        # Stiffness matrix

        # def analyticalStiffnessMatrix(mesh,u,ndof,idb,familySet,partialAreas,surfaceCorrection,
        #                               V,par_omega,c,model,damage,history):
        def analyticalStiffnessMatrix(mesh,u,bc,family,par_omega,penalty,mu=None):
            '''Generates the stiffness matrix for the quasi-static solver
            
            Input parameters:
            
            - mesh: mesh generated with Mesh 
            - u: displacement of the nodes | 2N
            - bc: boundary conditions generated with BoundaryConditions
            - family: family generated with Family
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
            - penalty: value to use in constrained dofs of the stiffness matrix
            - mu: damage factor

            Output parameters:
            
            - A: PD static matrix'''

            self._family=family                
            C=self.c[0]/2*pd.general_functions.weightedVolume(par_omega) 
            # penalty=1e10
            V=mesh.A
            N=len(mesh.points)
            A=np.zeros((2*N,2*N),dtype=float) # 2N GDLs
            # m=pd.general_functions.weightedVolume(par_omega)*np.ones((N))
            m=pd.general_functions.weightedVolume(par_omega)
            
            for ii in range(0,N):
                # dofi=[idb[2*ii,idb[2*ii+1]]]
                dofi=bc.dofi[ii]
                #family=familySet[ii,familySet[ii,:]>-1]
                # family=family.j[ii]
                try:
                    dofj=bc.dofj[ii] # (j,2)
                except AttributeError:
                    bc.compute_dofj(family)
                    dofj=bc.dofj[ii] # (j,2)
                eta=u[dofj]-u[dofi] # (j,2)
                xi=family.xj[ii]-mesh.points[ii] # (j,2)
                M=eta+xi # (j,2)
                normaj=np.linalg.norm(xi,axis=1) # (j,)
                norma_eta=np.linalg.norm(M,axis=1) # (j,)
                omegaj=pd.general_functions.influenceFunction(normaj,par_omega) # (j,)

                if type(mu)!=type(None): # damage factor was provided
                    µj=mu#[ii] # np.array of np.arrays ?
                else: # damage factor was not provided --> compute it
                    S=(np.linalg.norm(eta+xi,axis=1)-normaj)/normaj # Calculate stretch | (j,)
                    self.update_history_S(ii,S)
                    try:
                        #noFail=bc.noFail_ij[ii]
                        bc.noFail_ij#[ii]
                    except AttributeError:
                        bc.compute_noFail_ij(family)
                        #noFail=bc.noFail_ij[ii]
                    µj=damageFactor(self.history.S[ii],ii,family,bc)

                if dofi[0]<bc.ndof or dofi[1]<bc.ndof:
                    txxx=C*2./m*omegaj/norma_eta**2*\
                        family.PAj[ii]*family.SCj[ii]*µj*V # used for all (ti1u,ti2u,tj1u,tj2u,ti1v,ti2v,tj1v,tj2v)
                    ti2u_tj2u_ti1v_tj1v=M[:,0]*M[:,1] # used in ti2u, tj2u, ti1v and tj1v

                # if dofi[0]<=bc.ndof:
                if dofi[0]<bc.ndof:
                    # First dof of node ii is free
                    
                    # ti1u=C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,0]*M[:,0]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aii
                    # ti2u=C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,1]*M[:,0]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aip
                    # tj1u=-C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,0]*M[:,0]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aij
                    # tj2u=-C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,1]*M[:,0]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aijp



                    # ti1u=C*2./m*omegaj/norma_eta**2*family.PAj[ii]*family.SCj[ii]*µj*V*M[:,0]**2
                    # ti2u=C*2./m*omegaj/norma_eta**2*family.PAj[ii]*family.SCj[ii]*µj*V*M[:,0]*M[:,1]
                    # tj1u=-ti1u
                    # tj2u=-ti2u



                    tx1u=M[:,0]**2 # used in ti1u and tj1u

                    ti1u=txxx*tx1u # Aii
                    ti2u=txxx*ti2u_tj2u_ti1v_tj1v # Aip
                    tj1u=-ti1u # Aij | -txxx*tx1u
                    tj2u=-ti2u # Aijp | -txxx*tx2u

                    A[dofi[0],dofi[0]]=np.sum(ti1u)
                    # A[dofi[0],dofi[0]]=fsum(ti1u)
                    A[dofi[0],dofj[:,0]]=tj1u
                    A[dofi[0],dofi[1]]=np.sum(ti2u) ######## HEREEEEEEEEEE
                    # A[dofi[0],dofi[1]]=fsum(ti2u) ######## HEREEEEEEEEEE
                    A[dofi[0],dofj[:,1]]=tj2u
                else:
                    # Constraint nodes
                    A[dofi[0],dofi[0]]=-penalty

                # if dofi[1]<=bc.ndof:
                if dofi[1]<bc.ndof:
                    # V

                    # ti1v=C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,0]*M[:,1]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aii
                    # ti2v=C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,1]*M[:,1]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aip
                    # tj1v=-C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,0]*M[:,1]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aij
                    # tj2v=-C*(1./m[family.j[ii]]+1./m[ii])*omegaj/norma_eta**2*M[:,1]*M[:,1]*\
                    #     family.PAj[ii]*family.SCj[ii]*µj*V # Aijp


                    # ti1v=C*2./m*omegaj/norma_eta**2*family.PAj[ii]*family.SCj[ii]*µj*V*M[:,0]*M[:,1]
                    # ti2v=C*2./m*omegaj/norma_eta**2*family.PAj[ii]*family.SCj[ii]*µj*V*M[:,1]**2
                    # tj1v=-ti1v
                    # tj2v=-ti2v




                    tx2v=M[:,1]**2 # used in ti2v and tj2v

                    ti1v=txxx*ti2u_tj2u_ti1v_tj1v#M[:,1]*M[:,0]
                    ti2v=txxx*tx2v
                    tj1v=-ti1v
                    tj2v=-ti2v

                    A[dofi[1],dofi[0]]=np.sum(ti1v) ######## HEREEEEEEEEEE
                    # A[dofi[1],dofi[0]]=fsum(ti1v) ######## HEREEEEEEEEEE
                    A[dofi[1],dofj[:,0]]=tj1v
                    A[dofi[1],dofi[1]]=np.sum(ti2v)
                    # A[dofi[1],dofi[1]]=fsum(ti2v)
                    A[dofi[1],dofj[:,1]]=tj2v
                else:
                    # Constraint nodes
                    A[dofi[1],dofi[1]]=-penalty

                if ii==5:
                    pass
                
            A=-A
                
            return A
        self.analyticalStiffnessMatrix=analyticalStiffnessMatrix

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
            s=(np.linalg.norm(xi+eta,axis=1)-norma)/norma

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
