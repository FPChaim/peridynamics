'''Functions related to the model'''

import numpy as np

class History:
    '''History object for convenience'''        
    def __init__(self):
        '''- S: Stretch
        - theta (θ): Dilatation'''
        pass

def weightedVolume(par_omega:tuple[float|int,int,int]) -> float:
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

    - m_anl: weighted volume'''
    
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

# fscalar is used in ____________ --> doesn't seem to be used --> find out

# def fscalar(x,damage,noFail,ii):
#     '''Input parameters:
    
#     - x: S*mu ?
#     - damage: 
#     - noFail: 
#     - ii: index of the i-th node
    
#     Output parameters:
    
#     - ff: '''
#     if damage.damageOn==True:
#         # Damage dependent crack
#         alpha=damage.alpha
#         beta=damage.beta
#         gamma=damage.gamma
#         if damage.phi[ii]>alpha:
#             Sc=damage.S*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
#         else:
#             Sc=damage.Sc

#         # New formulation
#         ff=(x<Sc)*x
#         ff[noFail]=x[noFail]
#     else:
#         ff=x
            
#     return ff # N (might be equal or smaller than len(x))

######## MOVE js, jth and dilatation TO Model AND UPDATE getForce ########

# js and jth are used in dilatation
# dilatation changes according to the model

# def js(x,Sc):
#     '''Input parameters:
    
#     - x: 
#     - Sc: 
    
#     Output parameters:
    
#     - jj: '''
#     # If x < Sc, then js = 0
#     jj=(x>=Sc)*(x/Sc-1)**2/(1+(x/Sc)**2)
    
#     return jj

# def jth(x,thetac_p,thetac_m):
#     '''Input parameters:
    
#     - x: 
#     - thetac_p: 
#     - thetac_m: 
    
#     Output parameters:
    
#     - jj: '''
    
#     # % If x < Sc, then js = 0
#     jj=(x>=thetac_p)*(x/thetac_p-1)**4/(1+(x/thetac_p)**5)+(x<=-thetac_m-1)**4/(1+(-x/thetac_m)**5)
    
#     return jj

# def dilatation(mesh,u,family,transvList,
#                idb,par_omega,c,model,damage=None,history=None,dt=None):
#     '''Input parameters:
    
#     - mesh: peridynamic mesh generated with Mesh
#     - u: 
#     - family: famnily generated with Family
#     - transvList: range of nodes
#     - idb: 
#     - par_omega: 
#     - c: 
#     - model: model generated with Model
#     - damage: 
#     - history: 
#     - dt: 
    
#     Output parameters:
    
#     - theta: dilatation (θ)
#     - history_thetaUp: '''
    
#     #from numpy import np.size,np.arange,np.zeros,np.pi,np.dot,np.ones,np.logical_or,np.hstack
#     #from numpy.linalg import np.linalg.norm

#     x=mesh.points
    
#     horizon=par_omega[0]
#     thetac_p=0.01 # For Lipton's damage
#     thetac_m=0.01 # For Lipton's damage
#     if np.size(transvList)==0: # Not a sepecific range of nodes was chosen
#         transvList=np.arange(0,len(x))
#     try:
#         history_theta=history.theta
#     except NameError:
#         history_theta=[]
#     theta=np.zeros((len(transvList))) # Initialize dilatation vector
#     transv_ind=0 # Introduced so that we can pass as argument a smaller matrix

#     for ii in transvList:
#         dofi=[idb[2*ii],idb[2*ii+1]]
#         #familySet=family[transv_ind,family[transv_ind,:]>-1]
#         familySet=family.j[transv_ind]
#         jj=familySet
#         xi=x[jj,:]-x[ii,:]
#         neigh_ind=np.arange(0,len(jj))
#         dofj=np.hstack((idb[2*jj].reshape((-1,1)),idb[2*jj+1].reshape((-1,1))))
#         eta=u[dofj]-u[dofi]
#         norma=np.linalg.norm(xi,axis=1)
        
#         if model.name=='Lipton Free Damage':
#             V_delta=np.pi*horizon**2
#             S_linear=np.dot(xi,eta)/norma**2
#             #theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
#                 #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
#             theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
#                 family.PAj[transv_ind]*family.SCj[transv_ind]
#             if damage!=None and history!=None and dt!=None:
#                 if damage.damageOn==True:
#                     wholeBonds=~damage.brokenBonds[ii,neigh_ind]
#                 else:
#                     wholeBonds=np.ones(theta_vec.shape)
    
#                 if model.dilatHt==False:
#                     theta[transv_ind]=np.sum(theta_vec*wholeBonds)
#                 else:
#                     historyS=history.S[ii,neigh_ind]
#                     history_upS=historyS+js(S_linear,damage.Sc)*dt
#                     XX=[history_upS, history.theta[ii]*np.ones((len(history.theta[jj]),1)),history.theta[jj]]
#                     # noFail is true if node ii or jj is in the no fail zone
#                     noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#                     H=damageFactor(XX,ii,np.arange(0,len(jj)),damage,noFail,model)
#                     theta[transv_ind]=np.sum(theta_vec*H[:,0]) # Tulio's model
#             else:
#                 theta[transv_ind]=np.sum(theta_vec)
        
#         elif model.name=='LPS 2D':
#             nu=c[2]
#             elong=np.linalg.norm(xi+eta,axis=1)-norma#-np.linalg.norm(xi)
#             S=elong/norma
#             #theta_vec=2*(2*nu-1)/(nu-1)/m*influenceFunction(norma,par_omega)*norma*elong*\
#                 #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
#             theta_vec=2*(2*nu-1)/(nu-1)/m*influenceFunction(norma,par_omega)*norma*elong*\
#                 family.PAj[transv_ind]*family.SCj[transv_ind]
#             if damage!=None and history!=None and dt!=None:
#                 historyS=history.S[ii,neigh_ind]
#                 S_max=historyS
#                 historyS[S>S_max]=S[S>S_max]
#                 # noFail is true if node ii or jj is in the no fail zone
#                 noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#                 # Evaluating the damage factor
#                 # If noFail is true then we will always have mu as one
#                 mu=damageFactor(S_max,ii,np.arange(0,len(jj)),damage,noFail,model)
#                 theta[transv_ind]=np.sum(theta_vec*mu) # Tulio's model
#             else:
#                 theta[transv_ind]=np.sum(theta_vec)
                
#         elif model.name=='LSJT':
#             V_delta=np.pi*horizon**2
#             S_linear=np.dot(xi,eta)/norma**2
#             #theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
#                 #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
#             theta_vec=1/V_delta*influenceFunction(norma,par_omega)*norma**2*S_linear*\
#                 family.PAj[transv_ind]*family.SCj[transv_ind]
#             if damage!=None and history!=None and dt!=None:
#                 historyS=history.S[ii,neigh_ind]
#                 history_upS=historyS+js(S_linear,damage.Sc)*dt
#                 XX=[history_upS, history.theta[ii]*np.ones((len(history.theta[jj]),1)),history.theta[jj]]
#                 # noFail is true if node ii or jj is in the no fail zone
#                 noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#                 H=damageFactor(XX,ii,np.arange(0,len(jj)),damage,noFail,model)
#                 theta[transv_ind]=np.sum(theta_vec*H[:,0]) # Tulio's model
#             else:
#                 theta[transv_ind]=np.sum(theta_vec)
                
#         elif model.name=='Linearized LPS':
#             #theta_vec=3/m*influenceFunction(norma,par_omega)*np.dot(eta,xi)*\
#                 #partialAreas[transv_ind,neigh_ind]*surfaceCorrection[transv_ind,neigh_ind]
#             theta_vec=3/m*influenceFunction(norma,par_omega)*np.dot(eta,xi)*\
#                 family.PAj[transv_ind]*family.SCj[transv_ind]
#             theta[transv_ind]=np.sum(theta_vec)
        
#         history_thetaUp=np.zeros(len(transvList),1) # Prellocating memory for theta up
#         if damage!=None and history!=None and dt!=None:
#             # Update integral of dilatation of x_i for this specific interaction
#             history_thetaUp[transv_ind]=history_theta[transv_ind]+jth(theta[transv_ind],thetac_p,thetac_m)*dt
#         transv_ind=transv_ind+1
        
#     return theta,history_thetaUp

# antiderivative, antiderivativePMB, f_potential and g_potential (g_potential doesn't seem to be used or was not implemented, but should belong there anmyway) are used in strainEnergyDensity

# def antiderivative(x,damage,noFail,ii):
    
#     # Modified PMB model
#     if damage.damageOn==True:
#         # Damage dependent crack
#         alpha=damage.alpha
#         beta=damage.beta
#         gamma=damage.gamma
#         if damage.phi[ii]>alpha:
#             Sc=damage.Sc*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
#         else:
#             Sc=damage.Sc
#         S0=[-0.98,0.95*Sc] # S0- and S0+
#         S1=[-0.99,1.05*Sc] # S1- and S1+
#         # Evaluate integration constants
#         A=np.array([[1,0,0,0,0],
#                  [0,1,0,0,0],
#                  [1,0,0,-1,0],
#                  [0,0,1,0,0],
#                  [0,0,-1,0,1]])
#         b=np.array([[S0[0]**2/2-S0[0]/(S0[0]-S1[0])*(S0[0]**2/2-S1[0]*S0[0])],
#                  [0],
#                  [S0[0]/(S0[0]-S1[0])*(S1[0]**2/2)],
#                  [S0[1]**2/2-S0[1]/(S1[1]-S0[1])*(-S0[1]**2/2+S1[1]*S0[1])],
#                  [S0[1]/(S1[1]-S0[1])*S1[1]**2/2]])
#         C=np.linalg.lstsq(A,b,rcond=None)[0] # least-squares solution to a linear matrix equation
#         p=(x<=S1[0])*C[3]+(x<=S0[0])*(x>S1[0])*(S0[0]/(S0[0]-S1[0])*(x**2/2-S1[0]*x)+C[0])\
#             +(x<=S0[1])*(x>S0[0])*(x**2/2+C[1])+(x<=S1[1])*(x>S0[1])*(S0[1]/(S1[1]-S0[1])\
#             *(S1[1]*x-x**2/2)+C[2])+(x>S1[1])*C[4]
#         # Correcting the noFail
#         p[noFail]=x[noFail]**2/2
#     else:
#         p=x**2/2
        
#     return p

# def antiderivativePMB(x,damage,noFail,ii):
#     # PMB model
#     if damage.damageOn==True:
#         # Damage dependent crack
#         alpha=damage.alpha
#         beta=damage.beta
#         gamma=damage.gamma
#         if damage.phi[ii]>alpha:
#             Sc=damage.Sc*min([gamma,1+beta*(damage.phi[ii]-alpha)/(1-damage.phi[ii])])
#         else:
#             Sc=damage.Sc
#         p=(x<Sc)*x**2/2
#         # Correcting the noFail
#         p[noFail]=x[noFail]**2/2
#     else:
#         p=x**2/2
    
#     return p

# def f_potential(S,norma,c,damage):
#     r1=damage.Sc*norma
#     #r1=3 # Uncomment for a better result
#     r2=r1
#     x=S*norma
#     if damage.damageOn==True:
#         ff=(x<=r1)*c[0]*x**2/2+(x>r2)*x
#     else:
#         ff=c[0]*x**2
        
#     return ff

# def g_potential(x,c,damage):
#     r1=damage.thetaC
#     r2=r1
#     if damage.damageOn==True:
#         gg=(x<=r1)*c[1]*x**2/2+(x>r2)*x
#     else:
#         gg=c[1]*x**2/2

#     return gg

# def strainEnergyDensity(x,u,theta,family,ii,idb,par_omega,c,model,damage,historyS=None,historyT=None) -> float:
#     '''Input parametesr:
    
#     - mesh: peridynamic mesh generated with Mesh | Nx2 
#     - u: displacements | 2N
#     - theta: 
#     - family: family generated with Family
#     - ii: 
#     - idb: 
#     - par_omega ([δ,ωδ,γ]): influence function parameters
#         - horizon (δ): peridynamic horizon
#         - omega (ωδ): normalized function type
#             - 1: Exponential
#             - 2: Constant
#             - 3: Conical
#             - 4: Cubic polynomial
#             - 5: P5
#             - 6: P7
#             - 7: Singular
#         - gamma (γ): integer 
#     - c: 
#     - model: model generated with Model
#     - damage: 
#     - historyS: stretch history
#     - historyT: theta history
    
#     Output parameters:
    
#     -W: strain energy density of a bond'''

#     #from numpy import np.array,np.arange,np.hstack,np.ones,np.sum,np.linalg.dot,pi,all,logical_or
#     #from numpy.linalg import norm
    
#     #W=0
#     #familySet=family[family>-1] # neigh
#     #familySet=family.compressed() # neigh
#     familySet=family.j[ii] # neigh
#     partialAreas=family.PAj[ii]
#     surfaceCorrection=family.SCj[ii]
#     dofi=np.array([idb[2*ii],idb[2*ii+1]]) # 2
#     m=weightedVolume(par_omega)
#     horizon=par_omega[0]
#     # Evaluate dilatation
#     if model.dilatation==True:
#         #theta_i=dilatation(x,u,family,partialAreas,ii,idb,m,par_omega,c,model)
#         theta_i=theta[ii]
#     neigh_ind=np.arange(0,len(familySet)) # neigh
#     jj=familySet # neigh
#     xi=x[jj,:]-x[ii,:] # neigh x 2
#     # dofj=np.array([idb[2*jj],idb[2*jj+1]]).reshape(-1,2) # neigh x 2
#     dofj=np.hstack((idb[2*jj].reshape(-1,1),idb[2*jj+1].reshape(-1,1))) # neigh x 2
    
#     eta=u[dofj]-u[dofi]
#     norma=np.linalg.norm(xi,axis=1)
#     s=(np.linalg.norm(xi+eta,axis=1)-norma)/norma
    
#     if model.name=='PMB DTT':
#         if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
#             noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#             mu=damageFactor(historyS[neigh_ind],ii,neigh_ind,damage,noFail,model) # NoFail not required
#             p=antiderivative(s,damage,noFail,ii)
#         else:
#             mu=np.ones((len(jj)))
#             p=antiderivative(s,damage,False,ii)
#         w=1/2*c[0]*influenceFunction(norma,par_omega)*norma**2*p*mu
#         #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
#         #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
#         W=np.sum(w*partialAreas*surfaceCorrection)
    
#     elif model.name=='Linearized LPS bond-based':
#         extension=np.linalg.dot(eta,xi)/norma
#         w=1/2*c[0]*influenceFunction(norma,par_omega)*extension**2/2
#         #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
#         #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
#         W=np.sum(w*partialAreas*surfaceCorrection)
        
#     elif model.name=='Linearized LPS':
#         kappa=c[0]*m/2+c[1]*m/3
#         alpha=c[1]
#         w=alpha/2*influenceFunction(norma,par_omega)*norma**2*(np.dot(eta,xi)/norma**2-theta_i/3)**2
#         #W=W+w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind]
#         #W=w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind]
#         #W=w*partialAreas.compressed()*surfaceCorrection.compressed()
#         W=w*partialAreas*surfaceCorrection
# #         if b_familyEnd:
# #             W=W+kappa*theta_i**2/2
       
#     elif model.name=='Lipton Free Damage' or model.name=='LSJ-T':
#         if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
#             noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#             XX=[historyS[neigh_ind],historyT[ii]*np.ones((len(jj))),historyT[jj]]
#             H=damageFactor(XX,ii,neigh_ind,damage,noFail,model)
#         else:
#             H=np.ones((len(jj),3))
#         Slin=np.linalg.dot(xi,eta)/norma**2
#         V_delta=np.pi*horizon**2
#         w=1/V_delta*(influenceFunction(norma,par_omega)*norma/horizon*H[:,0]*f_potential(Slin,norma**0.5,c,damage))
#         #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
#         #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
#         W=np.sum(w*partialAreas*surfaceCorrection)
        
#     elif model.name=='LPS 2D':
#         noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#         if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:      
#             mu=damageFactor(historyS(neigh_ind),ii,neigh_ind,damage,noFail,model) # NoFail not required
#         else:
#             mu=np.ones((len(jj)))
#         p=antiderivative(s,damage,noFail,ii)
#         #elong=norm(xi+eta)-norma
#         nu=c[2]
#         #w=c[1]/2*influenceFunction(norma,par_omega)*elong**2
#         w=c[1]/2*influenceFunction(norma,par_omega)*norma**2*(2*p)*mu
#         #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
#         #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
#         W=np.sum(w*partialAreas*surfaceCorrection)+(c[0]/2+c[1]*m/3*(1/6-(nu-1)/(2*nu-1)))*theta_i**2
    
#     elif model.name=='PMB':
#         if all(historyS)!=None and all(historyT)!=None and damage.damageOn==True:
#             noFail=np.logical_or(damage.noFail[ii],damage.noFail[jj])
#             mu=damageFactor(historyS[neigh_ind],ii,neigh_ind,damage,noFail,model) # NoFail not required
#             p=antiderivativePMB(s,damage,noFail,ii)
#         else:
#             mu=np.ones((len(jj)))
#             p=antiderivativePMB(s,damage,False,ii)
#         w=1/2*c[0]*influenceFunction(norma,par_omega)*norma**2*p*mu
#         #W=np.sum(w*partialAreas[neigh_ind]*surfaceCorrection[neigh_ind])
#         #W=np.sum(w*partialAreas.compressed()*surfaceCorrection.compressed())
#         W=np.sum(w*partialAreas*surfaceCorrection)
        
#     else:
#         raise NameError('Model not implemented.')
        
#     return W
