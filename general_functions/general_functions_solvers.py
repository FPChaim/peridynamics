'''Functions related to the solvers'''

from timeit import default_timer
import numpy as np
from scipy.interpolate import interp1d
import peridynamics as pd


######################################################################################################################################################
# SOLVER RELATED STUFF |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
######################################################################################################################################################

# checkBondCrack is used on quasi-static solver and dynamic solver

def checkBondCrack(x1,x2,x3,x4):
    '''Checks if the bond is crossing a specific line
    
    Input parameters:
    
    - x1: [x y]: is the initial node
    - x2: [x y]: is the final node
    - x3: [x,y]: is the initial point of the crack
    - x4: [x,y]: is the final point of the crack
    
    Output parameters:
    
    - P: [Px Py]: is the intersection point
    - check: True or False: is True if the intersection point belong to both the bond and to the crack segment'''
    
    #from numpy import np.inf,nan,np.errstate,np.array,np.isfinite,np.isnan,np.isinf
    #from numpy.linalg import det,norm

    with np.errstate(divide='ignore',invalid='ignore'):
    
        den=np.linalg.det([[np.linalg.det([[x1[0],1],[x2[0],1]]),np.linalg.det([[x1[1],1],[x2[1],1]])],
                [np.linalg.det([[x3[0],1],[x4[0],1]]),np.linalg.det([[x3[1],1],[x4[1],1]])]])
        
        # X-component
        det11=np.linalg.det([[x1[0],x1[1]],
                [x2[0],x2[1]]])
        det12=np.linalg.det([[x1[0],1],
                [x2[0],1]])
        det21=np.linalg.det([[x3[0],x3[1]],
                [x4[0],x4[1]]])
        det22=np.linalg.det([[x3[0],1],
                [x4[0],1]])
        # with np.errstate(divide='ignore',invalid='ignore'):
        Px=np.linalg.det([[det11,det12],
                [det21,det22]])/den
        
        # Y-component
        # det11=det([[x1[0],x1[1]],[x2[0],x2[1]]])
        det12=np.linalg.det([[x1[1],1],
                [x2[1],1]])
        # det21=det([[x3[0],x3[1]],[x4[0],x4[1]]])
        det22=np.linalg.det([[x3[1],1],
                [x4[1],1]])
        # with np.errstate(divide='ignore',invalid='ignore'):
        Py=np.linalg.det([[det11,det12],
                [det21,det22]])/den
        
        P=np.array([Px,Py])
        
        if np.isfinite(np.linalg.norm(P)) and ~np.isnan(np.linalg.norm(P)):
            # Check if P belongs to both the crack segment and the bond
            if P[0]>min([x3[0],x4[0]])-1e-12 and P[0]<max([x3[0],x4[0]])+1e-12 and \
                P[1]>min([x3[1],x4[1]])-1e-12 and P[1]<max([x3[1],x4[1]])+1e-12:
                # Intersection belong to the crack segment
                if P[0]>min([x1[0],x2[0]])-1e-12 and P[0]<max([x1[0],x2[0]])+1e-12 and \
                    P[1]>min([x1[1],x2[1]])-1e-12 and P[1]<max([x1[1],x2[1]])+1e-12:
                    # Intersection point belong to the bond:
                    check=True
                else:
                    # Intersection doesn't belong to the bond
                    check=False
            else:
                # Intersection doesn't belong to the crack segment
                check=False
        else:
            # np.infinite norm means parallel lines
            a1=(x2[1]-x1[1])/(x2[0]-x1[0])
            a2=(x4[1]-x3[1])/(x4[0]-x3[0]) # Should be equal
            
            if np.isinf(a1):
                # Invert the definition
                a1=(x2[0]-x1[0])/(x2[1]-x1[1])
                b1=x1[0]-a1*x1[1]
                xx3=a1*x3[1]+b1
                if xx3>x3[0]-1e-12 and xx3<x3[0]+1e-12:
                    # xx3 = x3[0] -> coincident lines
                    if x3[1]>=min([x1[1],x2[1]]) and x3[1]<=max([x1[1],x2[1]]) or\
                        x4[1]>=min([x1[1],x2[1]]) and x4[1]<=max([x1[1],x2[1]]) or\
                        min([x1[1],x2[1]])>=min([x3[1],x4[1]]) and max([x1[1],x2[1]])<=max([x3[1],x4[1]]):
                        # Collinear segments that intercept each other
                        check=True
                    else:
                        # Collinear segments that do not intercept each other
                        check=False
                else:
                    # Parallel lines
                    check=False
            else:
                b1=x1[1]-a1*x1[0]
                y3=a1*x3[0]+b1
                if y3>x3[1]-1e-12 and y3<x3[1]+1e-12:
                    # y3 = x3[1] -> coincident lines
                    if (x3[0]>=min([x1[0],x2[0]])) and x3[0]<=max([x1[0],x2[0]]) or\
                        x4[0]>=min([x1[0],x2[0]]) and x4[0]<=max([x1[0],x2[0]]) or\
                        min([x1[0],x2[0]])>=min([x3[0],x4[0]]) and max([x1[0],x2[0]])<=max([x3[0],x4[0]]):
                        # Collinear segments that intercept each other
                        check=True
                    else:
                        # Collinear segments that do not intercept each other
                        check=False
                else:
                    # Parallel lines
                    check=False
                
    return P,check

# initialU0 was used in the quasi-static solver --> changed to a one liner

# def initialU0(N,n_tot):#,bc_set
#     '''Input parameters:
    
#     - N = length(idb) --> idb --> output of boundaryCondition | % N= 2*nn;
#     - n_tot: Number of load steps | type: int --> solver_QuasiStatic(x,n_tot,idb,b,bc_set,family,partialAreas,surfaceCorrection,T,c,model,par_omega,ndof,V,damage,history,noFailZone)
#     - bc_set: contains the set of constrained degrees of freedom on the first collumn and its corresponding value on the second collumn; the third collumn contains the corresponding dof velocity <-- boundaryCondition
    
#     Output parameters:
    
#     - un: initial displacement vector (filled with zeros) | 2Nxn_tot'''
    
#     from numpy import zeros

#     #n_con=len(bc_set)
#     un=zeros((N,n_tot))
#     # un[-n_con:,0]=bc_set[:,1]/n_tot

#     return un

# getForce is used on the quasi-static solver

def getForce(mesh,u,bn,bc,family,bc_setn,V,par_omega,model,penalty):
    '''Input Parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - u: displacements (trial until convergence) | 2N
    - bn: partial load (from the quasi-static solver) | 2N
    - bc: bonundary conditions generated with BoundaryConditions
    - family: family generated with Family
    - bc_setn: bc_set for the given load step
    - V: element area
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
    - model: model generated with Model
    - penalty: value to use in constrained dofs of the stiffness matrix
    
    Output parameters: 
    
    - f: vector state force between j and i nodes
    - phi: damage index
    - f_model: internal force only for all points (including b)
    - Also updates model.history_S'''
        
    N=len(mesh.points)
    f=np.zeros(len(u)) # 2N
    # Evaluate dilatation
    if model.dilatation==True:
        # theta,history.theta=model.dilatation(mesh,u,family.j,family.PAj,family.SCj,
        #                                  [],bc.idb,par_omega,model,damage,history,0) ####################################################################################### <-- NEED IMPLEMENTATION
        pass
    # Evaluate the force, history variables and damage
    phi=np.zeros((len(mesh.points)))
    for ii in range(0,N):
        #family=family[ii,family[ii,:]>-1] # Neighbours node list
        #family=family[ii].compressed() # Neighbours node list
        dofi=bc.dofi[ii] # (2,)
        # jj=family.j[ii]
        # neigh_index=np.arange(0,len(jj))
        # noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
        # noFail=bc.noFail_ij[ii]
        if model.dilatation==True:
            # fij,mu_j=model.T(mesh,u,theta,ii,family,bc) # (neigh,2) | updates model.history_S and model.histoy_theta ####################################################################################### <-- NEED IMPLEMENTATION
            pass 
        else:    
            fij,mu_j=model.T(mesh,u,ii,family,bc) # (neigh,2) | updates model.history_S
            
        # fij: Nneighx2 | Vj: Nneigh | _lambda: Nneigh <-- f_i: 2
        f_i=np.sum(fij*((family.PAj[ii]*family.SCj[ii]).reshape(-1,1)),axis=0)
        f[dofi]=f_i
        # Damage index
        areaTot=np.sum(family.PAj[ii]) # (1,)
        partialDamage=np.sum(mu_j*family.PAj[ii]) # (1,)
        phi[ii]=1-partialDamage/areaTot # (1,)
    
    f_model=f # Internal force only for all points (including b)
    f=(f+bn)*V
    # Change it to add boundary conditions
    # penalty=1e10
    
    if bc_setn.size>0:
        # The second collumn of bc_set represents the value of the constrain; minus for the relation u = -K\f
        f[bc.ndof:]=-penalty*np.zeros(len(bc_setn[:,1]))
        # f[bc.ndof:]=-penalty
        
    return f,phi,f_model

# Energy is instantiated and used in the solver quasi-static and dynamic

class Energy:
    #'''Class so that one can initiate the energy object with built-in attributes that will be used throughout the script'''
    '''Energy object for convenience'''
    def __init__(self):
        '''- W: Potential Energy
        - KE: Kinectic Energy
        - EW: External Work'''
        pass

# tangentStiffnessMatrix is used on the quasi-static solver

def tangentStiffnessMatrix(mesh,u,family,bc,par_omega,model,penalty):
    '''Implementation of quasi-static solver
    
    Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - u: | 2N
    - family: family generated with Family
    - bc: boundary conditions generated with BoundaryConditions
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
    - model: model generated with Model
    
    Output parameters: 
    
    - K: tangent stiffness matrix | 2Nx2N'''
    
    #from numpy import zeros,np.arange,floor,np.hstack,np.sum
    #from timeit import default_timer
    
    x=mesh.points
    h=mesh.h # Nodal spacing
    A=mesh.A
    epsilon=h*1e-7
    N=len(u) # 2N
    epsilon_vector=np.zeros((N)) # 2N
    # Defining the node's degree of freedom index
    dof_vec=np.zeros(x.shape,dtype=int) # Nx2
    # for kk in range(0,len(x)):
    #     dof_vec[kk]=[idb[2*kk],idb[2*kk+1]]
    dof_vec=bc.dofi
    # Initialize the tangent stiffness matrix to zero
    K=np.zeros((N,N))
    # Transverse each node in the discretization
    tic=default_timer()
    for ii in range(0,len(x)):
        #transvList=[ii, family[ii,family[ii,:]>-1]] # Node i and all neighbours of node i
        #transvList=family[ii].compressed() # Node i and all neighbours of node i
        #transvList=np.hstack((ii,family[ii].compressed())) # Node i and all neighbours of node i
        # transvList=family.i_j[ii] # Node i and all neighbours of node i
        transvList=np.hstack((ii,family.j[ii])) # Node i and all neighbours of node i
        
        if model.dilatation==True: # The system has dilatation
            li=len(transvList)
            neighIndex=0
            #for jj in family[ii,family[ii,:]>-1]:
            #for jj in family[ii].compressed():
            for jj in family.j[ii]:
                #for ll in family[ii,family[ii,:]>-1]:
                #for ll in family[ii].compressed():
                for ll in family.j[ii]:
                    #if ~np.sum(transvList)==ll:
                    if np.sum(transvList)!=ll:
                        transvList[li+neighIndex]=ll
                        neighIndex=neighIndex+1
                        
        for jj in transvList:
            # Evaluate the force state at x1 under perturbations of displacement
            dofj=dof_vec[jj]
            for rr in dofj: # For each displacement degree of freedom of node jj: [2*jj] = e1 and 2*jj+1 = e2
                # rr: 1
                epsilon_vector[rr]=epsilon
                u_plus=u+epsilon_vector
                u_minus=u-epsilon_vector
                
                # Evaluate related dilatation
                theta_plus=np.zeros((len(x)))
                theta_minus=np.zeros((len(x)))
                if model.dilatation==True:
                    #transvListII=[ii,family[ii,family[ii,:]>-1]] # Transversal list of affected dilatations
                    #transvListII=np.hstack((ii,family[ii].compressed())) # Transversal list of affected dilatations
                    # transvListII=family.i_j[ii] # Transversal list of affected dilatations
                    transvListII=np.hstack((ii,family.j[ii])) # Transversal list of affected dilatations
                    # theta_plus[transvListII]=dilatation(mesh,u_plus,family.j[transvListII],family.PAj[transvListII],family.SCj[transvListII],transvListII,idb,par_omega,c,model) ####################################################################################### <-- NEED IMPLEMENTATION
                    # theta_minus[transvListII]=dilatation(mesh,u_minus,family.j[transvListII],family.PAj[transvListII],family.SCj[transvListII],transvListII,idb,par_omega,c,model) ####################################################################################### <-- NEED IMPLEMENTATION
                
                #kk=family[ii,family[ii,:]>-1]
                #kk=family[ii].compressed()
                kk=family.j[ii]
                neigh_index=np.arange(0,len(kk))
                dofi=dof_vec[ii] # 2
                if model.dilatation==True:
                    # T_plus=model.T(mesh,u_plus,theta_plus,ii,kk,dof_vec,par_omega,c,model,[],damage,0,history.S[ii,neigh_index],history.theta,[])[0] ####################################################################################### <-- NEED IMPLEMENTATION
                    # T_minus=model.T(mesh,u_minus,theta_minus,ii,kk,dof_vec,par_omega,c,model,[],damage,0,history.S[ii,neigh_index],history.theta,[])[0] ####################################################################################### <-- NEED IMPLEMENTATION
                    pass
                else:
                    T_plus=model.T(mesh,u_plus,ii,family,bc)[0]
                    T_minus=model.T(mesh,u_minus,ii,family,bc)[0]
                  
                #T_plus shape: N neigh x 2
                #T_minus shape: N neigh x 2
                
#                 from numpy import shape
#                 print(f'T_plus shape = {shape(T_plus)}')
#                 print(f'partialAreas[ii,neigh_index] shape = {shape(partialAreas[ii,neigh_index])}')
#                 print(f'ii = {ii}')
#                 print(f'neigh_index = {neigh_index}')
                
                    
                #f_plus_shape: N neigh x 2
                #f_minus_shape: N neigh x 2
                #f_plus=T_plus*partialAreas[ii,neigh_index].reshape((-1,1))*surfaceCorrection[ii,neigh_index].reshape((-1,1))*A # S_max set to zero
                f_plus=T_plus*family.PAj[ii].reshape((-1,1))*family.SCj[ii].reshape((-1,1))*A # S_max set to zero
                #f_minus=T_minus*partialAreas[ii,neigh_index].reshape((-1,1))*surfaceCorrection[ii,neigh_index].reshape((-1,1))*A # S_max set to zero again
                f_minus=T_minus*family.PAj[ii].reshape((-1,1))*family.SCj[ii].reshape((-1,1))*A # S_max set to zero again
                f_diff=np.sum(f_plus-f_minus,axis=0) # 2
                # For each displacement degree of freedom of node kk: (2*jj-1) = e1 and 2*jj = e2
                # shape dofi: 2, shape rr: 1
                # K[dofi,rr]=K[[line1,line2],col]
                K[dofi,rr]=f_diff/2/epsilon
                epsilon_vector[rr]=0 # Resetting to zero
        
        # if ii+1==np.floor(len(x)/4) or ii+1==np.floor(len(x)/2) or ii+1==3*np.floor(len(x)/4) or ii+1==len(x): # 25 %, 50 %, 75 % or 100 %
        toc=default_timer()
        eta=toc-tic
        
        pd._misc.progress_printer(values=[(ii+1)/len(x),eta/(ii+1)*(len(x)-(ii+1))],
                            fmt='Calculating the stiffness matrix... |%b| %p % | ETA = %v1 s',bar_sep=40,
                            hundred_percent_print=' Done.')
            
    # Adapting for the constrained dofs
    for ii in range(bc.ndof,len(u)):
        K[ii,:]=np.zeros((len(K[ii])))
        #K[ii,ii]=1e10 # Penalty
        K[ii,ii]=penalty # Penalty
        
    return K

#parFor_loop, fload and sampling are used in the dynamic solver

def parFor_loop(mesh,family,bc,model,u_n,ii,par_omega,theta,b_Weval):
    # Loop on the nodes
    if model.dilatation==True:
        fij,mu_j=model.T(mesh,u_n,theta,ii,family,bc)
    else:
        fij,mu_j=model.T(mesh,u_n,ii,family,bc)
    f_i=np.sum(fij*(family.PAj[ii]*family.SCj[ii]).reshape((-1,1)))
    # Damage index
    areaTot=np.sum(family.PAj[ii])
    partialDamage=np.sum(mu_j*family.PAj[ii])
    phi_up=1-partialDamage/areaTot
    if b_Weval==True:
        if model.dilatation==True:
            W=model.strainEnergyDensity(mesh,u_n,theta,family,ii,bc,par_omega)
        else:
            W=model.strainEnergyDensity(mesh,u_n,family,ii,bc,par_omega)
        # Stored strain energy
        energy_pot=W*mesh.A
    else:
        energy_pot=0

    return f_i,phi_up,energy_pot

# def forceSection(mesh,family,bc,model,u_n,ii,theta):
#     # Loop on neighbourhood
#     fij=

def fload(mesh,family,bc,model,u,theta):
    '''Calculates load'''
    F=np.array([0,0])
    const_dof=np.where(bc.idb>bc.ndof)
    if bc.bc_set.size>0:
        load_dof=const_dof[bc.bc_set[:,2]!=0]
        load_points=np.floor(load_dof/2)
        load_points=np.unique(load_points).astype(int)
        for kk in range(0,len(load_points)):
            ii=load_points[kk]
            if model.dilatation==True:
                fij=model.T(mesh,u,theta,ii,family,bc)[0]
            else:
                fij=model.T(mesh,u,ii,family,bc)[0]
            f_i=np.sum(fij*family.PAj[ii]*family.SCj[ii])
            F=F-f_i*mesh.A
    return F

def sampling(x,t,ts):
    '''Reducing the compoents'''
    xs=np.zeros((len(x),(len(ts))))
    for iii in range(0,len(x)):
        f=interp1d(t,x[iii])
        xs[iii]=f(ts)
    
    return xs
