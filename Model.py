'''Provides all the code related to a 2D peridynamic model'''

# from sys import float_info
import numpy as np
from peridynamics.general_functions import weightedVolume,influenceFunction,History
# from math import fsum

############ Temporary ###############

def interactionForce_LLPSBB():
    pass

def interactionForce_StateBased():
    pass

def interactionForce_LSJT():
    pass

def interactionForce_Lipton():
    pass

def interactionForce_PMBDTT():
    pass

############ Temporary ###############

class Model:
    '''Class so that one can initiate the model object with built-in attributes that will be used throughout the script'''
    def __init__(self,name,par_omega,E,G0,bc):
        '''- name: str
        - dilatation: boolean
        - dilatHt: boolean
        - linearity: boolean
        - stiffnessAnal: boolean'''

        #from numpy import np.array,np.pi

        δ=par_omega[0]
        # self.par_omega=par_omega

        if  name=='LBB':
            self.name='LBB'
            # Force vector state
            self.T=interactionForce_LLPSBB
            # Boolean parameters
            self.dilatation=False # True if the model has a dilatation term
            self.dilatHt=False # ?
            self.linearity=True # True if the model is linear with the displacement field
            self.stiffnessAnal=False # True if there is a analyticalStiffnessMatrix implemented
            # Micro modulus
            mm=weightedVolume(par_omega)
            self.c=np.array([6*E/mm])
            # Critical relative elongation
            bc.damage.Sc=G0/4
            # Damage parameters
            bc.damage.alpha=None # ?
            bc.damage.beta=None # ?
            bc.damage.gamma=None # ?
            # self.number=2
            
            def damageFactor(x,ii,family,bc):
                '''Input parameters:
                
                - x: stretch, js integral, jtheta integral ...
                - ii: index of node i
                - neighIndex: indices of nodes j (neighbors of node i)
                - damage: all needed data
                - noFail: true if one of the nodes is in a non fail zone
                - model: all needed data

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

            # Force vector state
            def T(mesh,u,ii,family,bc):
                '''Input parameters:
                
                - x: position of the nodes
                - u: degree of freedom displacement vector ? | displacement of the nodes ? | 2N (double len of mesh points)
                - ii: index of the i-th node
                - jj: indices of the j-ths nodes inside i's family
                - dof_vec: matrix with the degrees of freedom corresponding for each node | Nx2
                - par_omega: [horizon,omega,alpha]
                - c: micro-modulus
                - model: 
                - separatorDamage: doesn't do np.anything but is useful to separate the normal variables to the np.ones needed for damage simulation
                - damage: 
                - dt: step time | same as n_tot ?
                - history_S: N x maxNeigh matrix containing the stretch (S) of each node j in relation to i <-- history.S <-- historyDependency
                - noFail: set of nodes for which we have no fail condition (mu = 1 always) <-- boundaryCondition
                
                Output parameters:
                
                - f: vector state force between j and i nodes
                - history_S: maximum stretch for the given bond
                - mu: damage factor'''
                
                #from numpy import np.ones,np.arange,np.array
                #from numpy.linalg import np.linalg.norm
                
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
                f=self.c[0]*(influenceFunction(norma,par_omega)*norma*S*µ).reshape((-1,1))*ee # (neigh,) * (neigh,2) --> (neigh,2)
                
                return f,µ
            self.T=T

            def strainEnergyDensity(mesh,u,family,ii,bc,par_omega) -> float:
                '''Input parametesr:
                
                - x: | Nx2 
                - u: | 2N
                - theta: 
                - family: 
                - partialAreas: 
                - surfaceCorrection: 
                - ii: 
                - idb: 
                - par_omega: 
                - c: 
                - model: 
                - damage: 
                - historyS: 
                - historyT: 
                
                Output parameters:
                
                -W: strain energy density of a bond'''

                #from numpy import np.sum
                #from numpy.linalg import np.linalg.norm
                
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
                w=1/2*self.c[0]*influenceFunction(norma,par_omega)*norma**2*p*µ
                #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
                #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
                W=np.sum(w*family.PAj[ii]*family.SCj[ii])
                            
                return W
            self.strainEnergyDensity=strainEnergyDensity

        elif name=='LPS 2D':
            self.name='LPS 2D'
            # Force vector state
            self.T=interactionForce_StateBased
            # Boolean parameters
            self.dilatation=True
            self.dilatHt=False # ?
            self.linearity=False
            self.stiffnessAnal=False
            # Micro modulus

            # Critical relative elongation

            # Damage parameters - no damage dependent Sc
            bc.damage.alpha=0
            bc.damage.beta=0
            bc.damage.gamma=1
            # self.number=4

        elif name=='LSJ-T':
            self.name='LSJ-T'
            # Force vector state
            self.T=interactionForce_LSJT
            # Boolean parameters
            self.dilatation=True
            self.dilatHt=True
            self.linearity=True
            self.stiffnessAnal=False
            # Micro modulus

            # Critical relative elongation

            # Damage parameters
            bc.damage.alpha=None # ?
            bc.damage.beta=None # ?
            bc.damage.gamma=None # ?
            # self.number=6
            
        elif name=='Lipton Free Damage':
            self.name='Lipton Free Damage'
            # Force vector state
            self.T=interactionForce_Lipton
            # Boolean parameters
            self.dilatation=True
            self.dilatHt=False
            self.linearity=True
            self.stiffnessAnal=False
            # Micro modulus

            # Critical relative elongation

            # Damage parameters - damage dependent Sc
            bc.damage.alpha=0.2
            bc.damage.beta=0.2
            bc.damage.gamma=1.4
            # self.number=3
            
        elif name=='PMB':
            self.name='PMB'
            # Boolean parameters
            self.dilatation=False
            self.dilatHt=False # ?
            self.linearity=False
            self.stiffnessAnal=True
            # Micro modulus
            mm=weightedVolume(par_omega)
            self.c=np.array([6*E/mm])
            # Critical relative elongation
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

            # def damageFactor(self,x,ii,neighIndex,damage,noFail,model):
            def damageFactor(x,ii,family,bc):
                '''Input parameters:
                
                - x: stretch, js integral, jtheta integral ...
                - ii: index of node i
                - neighIndex: indices of nodes j (neighbors of node i)
                - damage: all needed data
                - noFail: true if one of the nodes is in a non fail zone
                - model: all needed data

                Output parameters:

                - µ: damage factor for every neighbot of ii'''
                
                #from numpy import np.ones,np.min,np.any

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

            # Force vector state
            def T(mesh,u,ii,family,bc):
                '''Input parameters:
                
                - x: position of the nodes
                - u: degree of freedom displacement vector ? | displacement of the nodes ? | 2N (double len of mesh points)
                - ii: index of the i-th node
                - jj: indices of the j-ths nodes inside i's family
                - dof_vec: matrix with the degrees of freedom corresponding for each node | Nx2
                - par_omega: [horizon,omega,alpha]
                - c: micro-modulus
                - model: 
                - separatorDamage: doesn't do np.anything but is useful to separate the normal variables to the np.ones needed for damage simulation
                - damage: 
                - dt: step time | same as n_tot ?
                - history_S: N x maxNeigh matrix containing the stretch (S) of each node j in relation to i <-- history.S <-- historyDependency
                - noFail: set of nodes for which we have no fail condition (mu = 1 always) <-- boundaryCondition
                
                Output parameters:
                
                - f: vector state force between j and i nodes
                - history_S: maximum stretch for the given bond
                - mu: damage factor'''
                
                #from numpy import np.ones,np.arange,np.array
                #from numpy.linalg import np.linalg.norm
                
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
                f=self.c[0]*(influenceFunction(norma,par_omega)*norma*ff*µ).reshape((-1,1))*ee # (neigh,) * (neigh,2) --> (neigh,2)
                
                return f,µ
            self.T=T

            ## Stiffness matrix

            # def analyticalStiffnessMatrix(mesh,u,ndof,idb,familySet,partialAreas,surfaceCorrection,
            #                               V,par_omega,c,model,damage,history):
            def analyticalStiffnessMatrix(mesh,u,bc,family,par_omega,penalty,mu=None):
                '''Generates the stiffness matrix for the quasi-static solver
                
                Input parameters:
                
                - x: position of the nodes
                - u: displacement of the nodes
                - ndof: number of degrees of freedom
                - horizon: peridynamic horizon of the simulation
                - familySet: index of every node j (column) inside i (line) node's family
                - partialArea: partial areas of node in j collumn to the ith node
                - V: scalar volume for each node
                - par_omega: horizon omega alpha
                - c: material's constant

                Output parameters:
                
                - A: PD static matrix'''
                
                #from numpy import np.zeros,np.ones,np.sum
                #from numpy.linalg import np.linalg.norm

                self._family=family                
                C=self.c[0]/2*weightedVolume(par_omega) 
                # penalty=1e10
                V=mesh.A
                N=len(mesh.points)
                A=np.zeros((2*N,2*N),dtype=float) # 2N GDLs
                # m=weightedVolume(par_omega)*np.ones((N))
                m=weightedVolume(par_omega)
                
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
                    omegaj=influenceFunction(normaj,par_omega) # (j,)

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
                
                - x: | Nx2 
                - u: | 2N
                - theta: 
                - family: 
                - partialAreas: 
                - surfaceCorrection: 
                - ii: 
                - idb: 
                - par_omega: 
                - c: 
                - model: 
                - damage: 
                - historyS: 
                - historyT: 
                
                Output parameters:
                
                -W: strain energy density of a bond'''

                #from numpy import np.sum
                #from numpy.linalg import np.linalg.norm
                
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
                w=1/2*self.c[0]*influenceFunction(norma,par_omega)*norma**2*p*µ
                #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
                #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
                W=np.sum(w*family.PAj[ii]*family.SCj[ii])
                          
                return W
            self.strainEnergyDensity=strainEnergyDensity

        elif name=='PMB DTT':
            self.name='PMB DTT'
            # Force vector state
            self.T=interactionForce_PMBDTT
            # Boolean parameters
            self.dilatation=False
            self.dilatHt=False # ?
            self.linearity=False
            self.stiffnessAnal=False
            # Micro modulus

            # Critical relative elongation

            # Damage parameters - damage dependent Sc
            bc.damage.alpha=0.2
            bc.damage.beta=0.2
            bc.damage.gamma=1.4
            # self.number=1

    def initialize_history(self,family) -> None:
        '''Initiate the history dependency variable.

        Input parameters:

        - mesh:
        - family: 
        - model:

        Output parameters:

        - history: 3D matrix with the variables'''
                
        self.history=History()
        
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

        try:
            self.history.S[i][S>self.history.S[i]]=S[S>self.history.S[i]]
        except AttributeError:
            self.initialize_history(self._family)
            self.history.S[i][S>self.history.S[i]]=S[S>self.history.S[i]]

    def update_history_theta(self,i,theta) -> None:

        try:
            self.history.theta[i][theta>self.history.theta[i]]=theta[theta>self.history.theta[i]]
        except AttributeError:
            self.initialize_history(self._family)
            self.history.theta[i][theta>self.history.theta[i]]=theta[theta>self.history.theta[i]]

        

# model=Model2()
# help(model)