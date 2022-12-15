import numpy as np
from peridynamics._misc import progress_printer
from timeit import default_timer

### USED IN THE MODEL ### see if also used in solver

def influenceFunction(norma:np.ndarray|list|tuple,par_omega:tuple[float,int,int]) -> np.ndarray:
    '''Weights the contribution of volume (3D) or area (2D) dependent properties and modulate non-local effects

    Input parameters:
    
    - norma (|ξ|): distance between (bonded) points
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

    - ω(|ξ|): influence function'''
    
    #from numpy import exp#,atleast_1d,np.array
    
    δ=par_omega[0]
    ωδ=par_omega[1]
    γ=par_omega[2]

    # if type(norma)==list:
    #     norma=np.array(norma,dtype=float)

    if ωδ==1:
        # Exponential
        l=δ/4.3
        ω=np.exp(-norma**2/l**2)
    elif ωδ==2:
        # Constant
        ω=1/norma**γ
    elif ωδ==3:
        # Conical
        ω=(1-norma/δ)/norma**γ
    elif ωδ==4:
        # Cubic polynomial
        ω=(1-3*(norma/δ)**2+2*(norma/δ)**3)/norma**γ
    elif ωδ==5:
        # P5
        ω=(1-10*(norma/δ)**3+15*(norma/δ)**4-6*(norma/δ)**5)/norma**γ
    elif ωδ==6:
        # P7
        ω=(1-35*(norma/δ)**4+84*(norma/δ)**5-70*(norma/δ)**6+20*(norma/δ)**7)/norma**γ
    elif ωδ==7:
        # Singular
        ω=(δ/norma-1)/norma**γ

    # ω=atleast_1d(ω)
        
    ω[ω<0]=0

    return ω # (N,)

def weightedVolume(par_omega:tuple[float,int,int]) -> float:
    '''Analytical 'm': m is equal inside all the domain, as if it was on the bulk.
    
    Input parameters:

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

    - m_anl - Weighted Volume'''
    
    # from numpy import np.pi

    # np.pi=3.141592653589793
    # np.pi=3.14159265358979323846264338327950288419716939937510
    # π=3.141592653589793
    
    δ=par_omega[0]
    ω=par_omega[1]
    γ=par_omega[2]
    
    if ω==1:
        l=δ/4.3
        m_anl=l**2*np.pi*(l**2-np.exp(-(δ**2/l**2))*(l**2+δ**2))
    elif ω==2:
        if γ==0:
            m_anl=np.pi*δ**4/2
        else:
            m_anl=2*np.pi*δ**3/3
    elif ω==3:
        if γ==0:
            m_anl=np.pi*δ**4/10
        else:
            m_anl=np.pi*δ**3/6
    elif ω==4:
        if γ==0:
            m_anl=np.pi*δ**4/14
        else:
            m_anl=2*np.pi*δ**3/15
    elif ω==5:
        if γ==0:
            m_anl=5*np.pi*δ**4/84
        else:
            m_anl=5*np.pi*δ**3/42
    elif ω==6:
        if γ==0:
            m_anl=7*np.pi*δ**4/132
        else:
            m_anl=np.pi*δ**3/3
    elif ω==7:
        if γ==0:
            m_anl=np.pi*δ**4/6
        else:
            m_anl=np.pi*δ**3/3
    
    return m_anl

def fscalar(x,damage,noFail,ii):
    '''Input parameters:
    
    - x: S*mu ?
    - damage: 
    - noFail: 
    - ii: index of the i-th node
    
    Output parameters:
    
    - ff: '''
    if damage.damageOn==True:
        # Damage dependent crack
        alpha=damage.alpha
        beta=damage.beta
        gamma=damage.gamma
        if damage.phi[ii]>alpha:
            Sc=damage.S*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
        else:
            Sc=damage.Sc

        # New formulation
        ff=(x<Sc)*x
        ff[noFail]=x[noFail]
    else:
        ff=x
            
    return ff # N (might be equal or smaller than len(x))

### USED IN THE SOLVER ###

# TO FIX WHEN REACHING SOLVER

def js(x,Sc):
    '''Input parameters:
    
    - x: 
    - Sc: 
    
    Output parameters:
    
    - jj: '''
    # If x < Sc, then js = 0
    jj=(x>=Sc)*(x/Sc-1)**2/(1+(x/Sc)**2)
    
    return jj

def jth(x,thetac_p,thetac_m):
    '''Input parameters:
    
    - x: 
    - thetac_p: 
    - thetac_m: 
    
    Output parameters:
    
    - jj: '''
    
    # % If x < Sc, then js = 0
    jj=(x>=thetac_p)*(x/thetac_p-1)**4/(1+(x/thetac_p)**5)+(x<=-thetac_m-1)**4/(1+(-x/thetac_m)**5)
    
    return jj

def dilatation(mesh,u,family,transvList,
               idb,par_omega,c,model,damage=None,history=None,dt=None):
    '''Input parameters:
    
    - x: 
    - u: 
    - family: 
    - partialAreas: 
    - surfaceCorrection: 
    - transvList: range of nodes ?
    - idb: 
    - par_omega: 
    - c: 
    - model: 
    - damage: 
    - history: 
    - dt: 
    
    Output parameters:
    
    - theta: dilatation (θ)
    - history_thetaUp: '''
    
    #from numpy import np.size,np.arange,np.zeros,np.pi,np.dot,np.ones,np.logical_or,np.hstack
    #from numpy.linalg import np.linalg.norm

    x=mesh.points
    
    horizon=par_omega[0]
    thetac_p=0.01 # For Lipton's damage
    thetac_m=0.01 # For Lipton's damage
    if np.size(transvList)==0: # Not a sepecific range of nodes was chosen
        transvList=np.arange(0,len(x))
    try:
        history_theta=history.theta
    except NameError:
        history_theta=[]
    theta=np.zeros((len(transvList))) # Initialize dilatation vector
    transv_ind=0 # Introduced so that we can pass as argument a smaller matrix

    for ii in transvList:
        dofi=[idb[2*ii],idb[2*ii+1]]
        #familySet=family[transv_ind,family[transv_ind,:]>-1]
        familySet=family.j[transv_ind]
        jj=familySet
        xi=x[jj,:]-x[ii,:]
        neigh_ind=np.arange(0,len(jj))
        dofj=np.hstack((idb[2*jj].reshape((-1,1)),idb[2*jj+1].reshape((-1,1))))
        eta=u[dofj]-u[dofi]
        norma=np.linalg.norm(xi,axis=1)
        
        if model.name=='Lipton Free Damage':
            V_delta=np.pi*horizon**2
            S_linear=np.dot(xi,eta)/norma**2
            #theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
                #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
            theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
                family.PAj[transv_ind]*family.SCj[transv_ind]
            if damage!=None and history!=None and dt!=None:
                if damage.damageOn==True:
                    wholeBonds=~damage.brokenBonds[ii,neigh_ind]
                else:
                    wholeBonds=np.ones(theta_vec.shape)
    
                if model.dilatHt==False:
                    theta[transv_ind]=np.sum(theta_vec*wholeBonds)
                else:
                    historyS=history.S[ii,neigh_ind]
                    history_upS=historyS+js(S_linear,damage.Sc)*dt
                    XX=[history_upS, history.theta[ii]*np.ones((len(history.theta[jj]),1)),history.theta[jj]]
                    # noFail is true if node ii or jj is in the no fail zone
                    noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
                    H=damageFactor(XX,ii,np.arange(0,len(jj)),damage,noFail,model)
                    theta[transv_ind]=np.sum(theta_vec*H[:,0]) # Tulio's model
            else:
                theta[transv_ind]=np.sum(theta_vec)
        
        elif model.name=='LPS 2D':
            nu=c[2]
            elong=np.linalg.norm(xi+eta,axis=1)-norma#-np.linalg.norm(xi)
            S=elong/norma
            #theta_vec=2*(2*nu-1)/(nu-1)/m*influenceFunction(norma,par_omega)*norma*elong*\
                #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
            theta_vec=2*(2*nu-1)/(nu-1)/m*influenceFunction(norma,par_omega)*norma*elong*\
                family.PAj[transv_ind]*family.SCj[transv_ind]
            if damage!=None and history!=None and dt!=None:
                historyS=history.S[ii,neigh_ind]
                S_max=historyS
                historyS[S>S_max]=S[S>S_max]
                # noFail is true if node ii or jj is in the no fail zone
                noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
                # Evaluating the damage factor
                # If noFail is true then we will always have mu as one
                mu=damageFactor(S_max,ii,np.arange(0,len(jj)),damage,noFail,model)
                theta[transv_ind]=np.sum(theta_vec*mu) # Tulio's model
            else:
                theta[transv_ind]=np.sum(theta_vec)
                
        elif model.name=='LSJT':
            V_delta=np.pi*horizon**2
            S_linear=np.dot(xi,eta)/norma**2
            #theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
                #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
            theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
                family.PAj[transv_ind]*family.SCj[transv_ind]
            if damage!=None and history!=None and dt!=None:
                historyS=history.S[ii,neigh_ind]
                history_upS=historyS+js(S_linear,damage.Sc)*dt
                XX=[history_upS, history.theta[ii]*np.ones((len(history.theta[jj]),1)),history.theta[jj]]
                # noFail is true if node ii or jj is in the no fail zone
                noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
                H=damageFactor(XX,ii,np.arange(0,len(jj)),damage,noFail,model)
                theta[transv_ind]=np.sum(theta_vec*H[:,0]) # Tulio's model
            else:
                theta[transv_ind]=np.sum(theta_vec)
                
        elif model.name=='Linearized LPS':
            #theta_vec=3/m*influenceFunction(norma,par_omega)*np.dot(eta,xi)*\
                #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
            theta_vec=3/m*influenceFunction(norma,par_omega)*np.dot(eta,xi)*\
                family.PAj[transv_ind]*family.SCj[transv_ind]
            theta[transv_ind]=np.sum(theta_vec)
        
        history_thetaUp=np.zeros(len(transvList),1) # Prellocating memory for theta up
        if damage!=None and history!=None and dt!=None:
            # Update integral of dilatation of x_i for this specific interaction
            history_thetaUp[transv_ind]=history_theta[transv_ind]+jth(theta[transv_ind],thetac_p,thetac_m)*dt
        transv_ind=transv_ind+1
        
    return theta,history_thetaUp

def getForce(mesh,u,b,bc,family,bc_set,V,par_omega,model,penalty):
    '''Input Parameters:
    
    - x:
    - u: 
    - T: 
    - b: | 2N
    - family: 
    - partialAreas: 
    - surfaceCorrection: 
    - dof_vec: 
    - idb: 
    - ndof: 
    - bc_set: 
    - V: 
    - par_omega: 
    - c: 
    - model: 
    - damage: 
    - history: 
    
    Output parameters: 
    
    - f: vector state force between j and i nodes
    - history: 
    - phi: 
    - f_model: '''
    
    #from numpy import np.zeros,np.arange,np.logical_or,np.sum
    
    N=len(mesh.points)
    f=np.zeros(len(u)) # 2N
    # Evaluate dilatation
    if model.dilatation==True:
        #theta,history.theta=dilatation(x,u,family,partialAreas,surfaceCorrection,
                                         #[],idb,par_omega,c,model,damage,history,0)
        theta,history.theta=model.dilatation(mesh,u,family.j,family.PAj,family.SCj,
                                         [],bc.idb,par_omega,model,damage,history,0)
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
            fij,mu_j=model.T(mesh,u,theta,ii,family,bc) # (neigh,2) | updates model.history_S and model.histoy_theta
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
    f=(f+b)*V # <----------- *V está adicionando uma dimensão
    # Change it to add boundary conditions
    # penalty=1e10
    
    if bc_set.size>0:
        # The second collumn of bc_set represents the value of the constrain; minus for the relation u = -K\f
        # f[bc.ndof:]=-penalty*np.zeros(len(bc_set[:,1]))
        f[bc.ndof:]=-penalty
        
    return f,phi,f_model

##################################################################################

def initialU0(N,n_tot):#,bc_set
    '''Input parameters:
    
    - N = length(idb) --> idb --> output of boundaryCondition | % N= 2*nn;
    - n_tot: Number of load steps | type: int --> solver_QuasiStatic(x,n_tot,idb,b,bc_set,family,partialAreas,surfaceCorrection,T,c,model,par_omega,ndof,V,damage,history,noFailZone)
    - bc_set: contains the set of constrained degrees of freedom on the first collumn and its corresponding value on the second collumn; the third collumn contains the corresponding dof velocity <-- boundaryCondition
    
    Output parameters:
    
    - un: ? | 2Nxn_tot'''
    
    from numpy import zeros

    #n_con=len(bc_set)
    un=zeros((N,n_tot))
    # un[-n_con:,0]=bc_set[:,1]/n_tot

    return un

# ideia --> remover esta função super simples, que só é usada uma vez dentro do solver quasistatic

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

# def tangentStiffnessMatrix(mesh,u,idb,family,partialAreas,surfaceCorrection,
#                            T,ndof,par_omega,c,model,damage,history):
def tangentStiffnessMatrix(mesh,u,family,bc,par_omega,model,penalty):
    '''Implementation of quasi-static solver
    
    Input parameters:
    
    - mesh: | Nx2
    - u: | 2N
    - idb: | 2N
    - family: | N x maxNeigh
    - partialAreas: | N x maxNeigh
    - surfaceCorrection: | N x maxNeigh
    - T:
    - ndof: | int≤2N
    - par_omega: 
    - c: 
    - model: 
    - damage: 
    - history: 
    
    Output parameters: 
    
    - K: | 2Nx2N'''
    
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
                    #theta_plus[transvListII]=dilatation(x,u_plus,family[transvListII,:],partialAreas[transvListII,:],surfaceCorrection[transvListII,:],transvListII,idb,par_omega,c,model)
                    theta_plus[transvListII]=dilatation(mesh,u_plus,family.j[transvListII],family.PAj[transvListII],family.SCj[transvListII],transvListII,idb,par_omega,c,model)
                    #theta_minus[transvListII]=dilatation(x,u_minus,family[transvListII,:],partialAreas[transvListII,:],surfaceCorrection[transvListII,:],transvListII,idb,par_omega,c,model)
                    theta_minus[transvListII]=dilatation(mesh,u_minus,family.j[transvListII],family.PAj[transvListII],family.SCj[transvListII],transvListII,idb,par_omega,c,model)
                
                #kk=family[ii,family[ii,:]>-1]
                #kk=family[ii].compressed()
                kk=family.j[ii]
                neigh_index=np.arange(0,len(kk))
                dofi=dof_vec[ii] # 2
                if model.dilatation==True:
                    T_plus=model.T(mesh,u_plus,theta_plus,ii,kk,dof_vec,par_omega,c,model,[],damage,0,history.S[ii,neigh_index],history.theta,[])[0]
                    T_minus=model.T(mesh,u_minus,theta_minus,ii,kk,dof_vec,par_omega,c,model,[],damage,0,history.S[ii,neigh_index],history.theta,[])[0]
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
        
        if ii+1==np.floor(len(x)/4) or ii+1==np.floor(len(x)/2) or ii+1==3*np.floor(len(x)/4) or ii+1==len(x): # 25 %, 50 %, 75 % or 100 %
            toc=default_timer()
            eta=toc-tic
            if ii+1==np.floor(len(x)/4):
                right_fill=len(str(eta)[:str(eta).find('.')])
            
            progress_printer(values=[(ii+1)/len(x),eta/(ii+1)*(len(x)-(ii+1))],
                             fmt='Calculating the stiffness matrix... |%b| %p % | ETA = %v s',bar_sep=40,
                             right_fill_for_value=right_fill,
                             hundred_percent_print='\nStiffness matrix done.')
            
    # Adapting for the constrained dofs
    for ii in range(bc.ndof,len(u)):
        K[ii,:]=np.zeros((len(K[ii])))
        #K[ii,ii]=1e10 # Penalty
        K[ii,ii]=penalty # Penalty
        
    return K

def antiderivative(x,damage,noFail,ii):
    
    #from numpy import np.array
    #from numpy.linalg import lstsq
    
    # Modified PMB model
    if damage.damageOn==True:
        # Damage dependent crack
        alpha=damage.alpha
        beta=damage.beta
        gamma=damage.gamma
        if damage.phi[ii]>alpha:
            Sc=damage.Sc*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
        else:
            Sc=damage.Sc
        S0=[-0.98,0.95*Sc] # S0- and S0+
        S1=[-0.99,1.05*Sc] # S1- and S1+
        # Evaluate integration constants
        A=np.array([[1,0,0,0,0],
                 [0,1,0,0,0],
                 [1,0,0,-1,0],
                 [0,0,1,0,0],
                 [0,0,-1,0,1]])
        b=np.array([[S0[0]**2/2-S0[0]/(S0[0]-S1[0])*(S0[0]**2/2-S1[0]*S0[0])],
                 [0],
                 [S0[0]/(S0[0]-S1[0])*(S1[0]**2/2)],
                 [S0[1]**2/2-S0[1]/(S1[1]-S0[1])*(-S0[1]**2/2+S1[1]*S0[1])],
                 [S0[1]/(S1[1]-S0[1])*S1[1]**2/2]])
        C=np.linalg.lstsq(A,b,rcond=None)[0] # least-squares solution to a linear matrix equation
        p=(x<=S1[0])*C[3]+(x<=S0[0])*(x>S1[0])*(S0[0]/(S0[0]-S1[0])*(x**2/2-S1[0]*x)+C[0])\
            +(x<=S0[1])*(x>S0[0])*(x**2/2+C[1])+(x<=S1[1])*(x>S0[1])*(S0[1]/(S1[1]-S0[1])\
            *(S1[1]*x-x**2/2)+C[2])+(x>S1[1])*C[4]
        # Correcting the noFail
        p[noFail]=x[noFail]**2/2
    else:
        p=x**2/2
        
    return p

def antiderivativePMB(x,damage,noFail,ii):
    # PMB model
    if damage.damageOn==True:
        # Damage dependent crack
        alpha=damage.alpha
        beta=damage.beta
        gamma=damage.gamma
        if damage.phi[ii]>alpha:
            Sc=damage.Sc*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
        else:
            Sc=damage.Sc
        p=(x<Sc)*x**2/2
        # Correcting the noFail
        p[noFail]=x[noFail]**2/2
    else:
        p=x**2/2
    
    return p

def f_potential(S,norma,c,damage):
    r1=damage.Sc*norma
    #r1=3 # Uncomment for a better result
    r2=r1
    x=S*norma
    if damage.damageOn==True:
        ff=(x<=r1)*c[0]*x**2/2+(x>r2)*x
    else:
        ff=c[0]*x**2
        
    return ff

def g_potential(x,c,damage):
    r1=damage.thetaC
    r2=r1
    if damage.damageOn==True:
        gg=(x<=r1)*c[1]*x**2/2+(x>r2)*x
    else:
        gg=c[1]*x**2/2

    return gg

# def strainEnergyDensity(x,u,theta,family,partialAreas,surfaceCorrection,
#                         ii,idb,par_omega,c,model,damage,historyS=None,historyT=None) -> float:
def strainEnergyDensity(x,u,theta,family,
                        ii,idb,par_omega,c,model,damage,historyS=None,historyT=None) -> float:
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

    #from numpy import np.array,np.arange,np.hstack,np.ones,np.sum,np.linalg.dot,pi,all,logical_or
    #from numpy.linalg import norm
    
    #W=0
    #familySet=family[family>-1] # neigh
    #familySet=family.compressed() # neigh
    familySet=family.j[ii] # neigh
    partialAreas=family.PAj[ii]
    surfaceCorrection=family.SCj[ii]
    dofi=np.array([idb[2*ii],idb[2*ii+1]]) # 2
    m=weightedVolume(par_omega)
    horizon=par_omega[0]
    # Evaluate dilatation
    if model.dilatation==True:
        #theta_i=dilatation(x,u,family,partialAreas,ii,idb,m,par_omega,c,model)
        theta_i=theta[ii]
    neigh_ind=np.arange(0,len(familySet)) # neigh
    jj=familySet # neigh
    xi=x[jj,:]-x[ii,:] # neigh x 2
    # dofj=np.array([idb[2*jj],idb[2*jj+1]]).reshape(-1,2) # neigh x 2
    dofj=np.hstack((idb[2*jj].reshape(-1,1),idb[2*jj+1].reshape(-1,1))) # neigh x 2
    
    eta=u[dofj]-u[dofi]
    norma=np.linalg.norm(xi,axis=1)
    s=(np.linalg.norm(xi+eta,axis=1)-norma)/norma
    
    if model.name=='PMB DTT':
        if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
            noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
            mu=damageFactor(historyS[neigh_ind],ii,neigh_ind,damage,noFail,model) # NoFail not required
            p=antiderivative(s,damage,noFail,ii)
        else:
            mu=np.ones((len(jj)))
            p=antiderivative(s,damage,False,ii)
        w=1/2*c[0]*influenceFunction(norma,par_omega)*norma**2*p*mu
        #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
        #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
        W=np.sum(w*partialAreas*surfaceCorrection)
    
    elif model.name=='Linearized LPS bond-based':
        extension=np.linalg.dot(eta,xi)/norma
        w=1/2*c[0]*influenceFunction(norma,par_omega)*extension**2/2
        #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
        #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
        W=np.sum(w*partialAreas*surfaceCorrection)
        
    elif model.name=='Linearized LPS':
        kappa=c[0]*m/2+c[1]*m/3
        alpha=c[1]
        w=alpha/2*influenceFunction(norma,par_omega)*norma**2*(np.dot(eta,xi)/norma**2-theta_i/3)**2
        #W=W+w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind]
        #W=w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind]
        #W=w*partialAreas.compressed()*surfaceCorrection.compressed()
        W=w*partialAreas*surfaceCorrection
#         if b_familyEnd:
#             W=W+kappa*theta_i**2/2
       
    elif model.name=='Lipton Free Damage' or model.name=='LSJ-T':
        if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
            noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
            XX=[historyS[neigh_ind],historyT[ii]*np.ones((len(jj))),historyT[jj]]
            H=damageFactor(XX,ii,neigh_ind,damage,noFail,model)
        else:
            H=np.ones((len(jj),3))
        Slin=np.linalg.dot(xi,eta)/norma**2
        V_delta=np.pi*horizon**2
        w=1/V_delta*(influenceFunction(norma,par_omega)*norma/horizon*H[:,0]*f_potential(Slin,norma**0.5,c,damage))
        #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
        #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
        W=np.sum(w*partialAreas*surfaceCorrection)
        
    elif model.name=='LPS 2D':
        noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
        if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:      
            mu=damageFactor(historyS(neigh_ind),ii,neigh_ind,damage,noFail,model) # NoFail not required
        else:
            mu=np.ones((len(jj)))
        p=antiderivative(s,damage,noFail,ii)
        #elong=norm(xi+eta)-norma
        nu=c[2]
        #w=c[1]/2*influenceFunction(norma,par_omega)*elong**2
        w=c[1]/2*influenceFunction(norma,par_omega)*norma**2*(2*p)*mu
        #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
        #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
        W=np.sum(w*partialAreas*surfaceCorrection)+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
    
    elif model.name=='PMB':
        if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
            noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
            mu=damageFactor(historyS[neigh_ind],ii,neigh_ind,damage,noFail,model) # NoFail not required
            p=antiderivativePMB(s,damage,noFail,ii)
        else:
            mu=np.ones((len(jj)))
            p=antiderivativePMB(s,damage,False,ii)
        w=1/2*c[0]*influenceFunction(norma,par_omega)*norma**2*p*mu
        #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
        #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
        W=np.sum(w*partialAreas*surfaceCorrection)
        
    else:
        raise NameError('Model not implemented.')
        
    return W

class Damage:
    '''Object with built-in attributes that will be used throughout the script'''
    def __init__(self) -> None:
        '''- alpha:
        - beta:
        - brokenBonds: ii: node, jj: neighbors, kk: crack segment i to i+1
        - crackIn: Nx2 np.array of crack segments, where N≥2
        - damage_dependent_Sc
        - damage_on: Boolean
        - gamma: 
        - noFail: 
        - phi: 
        - Sc: critical relative elongation
        - thetaC:'''
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

class Energy:
    #'''Class so that one can initiate the energy object with built-in attributes that will be used throughout the script'''
    '''Energy object for convenience'''
    def __init__(self):
        '''- W: Potential Energy
        - KE: Kinectic Energy
        - EW: External Work'''
        pass

class History:
    '''History object for convenience'''        
    def __init__(self):
        '''- S: Stretch
        - theta (θ): Dilatation'''
        pass