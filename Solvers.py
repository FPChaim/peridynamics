'''Quasi-static and dynamic solvers'''

from timeit import default_timer
import numpy as np
from scipy.interpolate import interp1d
import peridynamics as pd

def solver_QuasiStatic(mesh:pd.Mesh,n_tot:int,bc:pd.BoundaryConditions,family:pd.Family,model:pd.Model,par_omega:tuple[float|int,int,int],penalty:float|int) -> tuple[np.ndarray,float,pd.general_functions.Energy,np.ndarray]:

    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - n_tot: number of load steps
    - bc: boundary conditions genretade with BoundaryConditions
    - family: family generetade with Family
    - model: model generated with Model
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
    
    Output parameters:
    
    - un: displacements
    - r: residual
    - energy: 
        - W: macro-elastic energy
        - KE: kinect energy
        - EW: external work
    - phi: damage index'''

    x=mesh.points
    V=mesh.A
    ndof=bc.ndof
    idb=bc.idb
    bc_set=bc.bc_set
    b=bc.bb # bodyForce
    # noFailZone=bc.noFail
    # maxNeigh=family.maxNeigh
    dof_vec=bc.dofi # node's degree of freedom index

    N=len(bc.idb) # 2N
#     if size(V)==1:
#         V=V*ones((len(x),1))
    # Create V_DOF
    # V_DOF=np.zeros((2*len(x)))
    # for ii in range(0,len(x)):
    #     V_DOF[[2*ii,2*ii+1]]=V#[ii]
    # V_DOF=np.full((2*len(mesh.points)),V)
    V_DOF=V
    # penalty=1e10
    
    # Damage variables
    # Defining cracking trespassing matrix
    free_points=np.full((len(mesh.points)),False,dtype=bool)

    if bc.damage.brokenBonds is None:
        bc.initialize_brokenBonds(family)

    if bc.damage.crackIn is not None:
        crackSegments=len(bc.damage.crackIn) # At least 2

        # for ii in range(0,len(model.history_S)):
        for ii in range(0,len(mesh.points)):
            x_j=family.xj[ii]

            # brokenBonds=np.empty((len(x_j),crackSegments-1),dtype=bool)
            for kk in range(0,crackSegments-1):
                for jj in range(0,len(x_j)):
                    # brokenBonds[jj,kk]=checkBondCrack(x[ii],x_j[jj],bc.damage.crackIn[kk],bc.damage.crackIn[kk+1])[1]
                    bc.damage.brokenBonds[ii][jj]=pd.general_functions.checkBondCrack(x[ii],x_j[jj],bc.damage.crackIn[kk],bc.damage.crackIn[kk+1])[1]

            if np.all(bc.damage.brokenBonds[ii])==True:
                free_points[ii]=True
            
            # bc.damage.brokenBonds[ii,0:len(x_j),:]=brokenBonds # ii: node, jj: neighbors, kk: crack segment i to i+1

        print('Check for broken bonds done.')
    
    # No fail to damage variable
    bc.damage.noFail=bc.noFail # ???
    phi=np.zeros((len(mesh.points),n_tot),dtype=float) # Damage index
    
    # Step 1 - initialization

    #un=initialU0(N,n_tot)
    un=np.zeros((N,n_tot))
    energy=pd.general_functions.Energy()
    energy.W=np.zeros((len(mesh.points),n_tot))
    energy.KE=np.zeros((len(mesh.points),n_tot))
    energy.EW=np.zeros((len(mesh.points),n_tot))
    f_int=np.zeros((N,n_tot))

    free_dof=np.full((2*len(mesh.points)),False,dtype=bool)
    # index_free_points=np.where(free_points)[0]
    # for kk in index_free_points:
    #     free_dof[dof_vec[kk]]=True
    free_dof[dof_vec[free_points]]=True

    # Initialize history
    # try:
    #     model.history.S
    # except AttributeError:
    model.initialize_history(family)
    
    for n in range(0,n_tot):
        # Step 2 - update the load step n <- n + 1 and pseudo-time t and update the boundary conditions
        bn=b*((n+1)/n_tot) # Partial load | 2N
        if bc_set.size>0:
            bc_setn=np.column_stack((bc_set[:,0],bc_set[:,1]*((n+1)/n_tot))) # len(bc_set)x2
            
        # Step 2.5 - assign an initial guess to the trial displacement utrial (for example, utrial = un)
        u_trial=un[:,n]
        if bc_set.size>0:
            # u_trial[ndof:]=bc_setn[:,1]
            u_trial[ndof:]=bc_setn[:,1]
        # Step 3 - evaluate the residual vector, r, and residual r and determine the convergence critrion for the load step
        epsilon=1e-4
        bc.damage.phi=phi[:,n] # Accessing current damage situation
        #r_vec,history,phi[:,n],f_int[:,n]=getForce(x,u_trial,T,bn,family,partialAreas,surfaceCorrection,dof_vec,idb,ndof,
                                                     #bc_setn,V_DOF,par_omega,c,model,damage,history) # Update to include arbitrary displacement kinematic conditions
        r_vec,phi[:,n],f_int[:,n]=pd.general_functions.getForce(mesh,u_trial,bn,bc,family,bc_setn,V_DOF,par_omega,model,penalty) # Update to include arbitrary displacement kinematic conditions
        # r_max=epsilon*np.max([np.linalg.norm(bn[:ndof]*V_DOF[:ndof],ord=np.inf),np.linalg.norm(f_int[:ndof,n]*V_DOF[:ndof],ord=np.inf)]) # normalizing the maximum residual | (1,)
        r_max=epsilon*np.max([np.linalg.norm(bn[:ndof]*V_DOF,ord=np.inf),np.linalg.norm(f_int[:ndof,n]*V_DOF,ord=np.inf)]) # normalizing the maximum residual | (1,)
        
        if r_max==0: # No forces on the non-constrain nodes
            r_max=epsilon
        # Step 5 - Apply Newton's method to minimize the residual
        r=np.linalg.norm(r_vec[:ndof],ord=np.inf) # (1,)
        alpha=1
        
        # -------------------- Newton's method ----------------
        if model.linearity==False:
            # Suitable for non-linear models
            _iter=1
            while r>r_max: # force
                # Damage
                bc.damage.phi=phi[:,n] # Accessing current damage situation
                if model.stiffnessAnal==False:
                    #K=tangentStiffnessMatrix(mesh,u_trial,idb,family,partialAreas,surfaceCorrection,T,ndof,par_omega,c,model,damage,history)
                    K=pd.general_functions.tangentStiffnessMatrix(mesh,u_trial,family,bc,par_omega,model,penalty)
                else:
                    #K=analyticalStiffnessMatrix(mesh,u_trial,ndof,idb,family,partialAreas,surfaceCorrection,V,par_omega,c,model,damage,history)
                    K=model.analyticalStiffnessMatrix(mesh,u_trial,bc,family,par_omega,penalty)
                # print('Stiffness matrix done')
                # K: 2Nx2N
                # r_vec: 2N
                # du=np.linalg.lstsq(-K[np.ix_(~free_dof,~free_dof)],r_vec[~free_dof],rcond=None)[0] # 2N
                du=np.linalg.solve(-K[np.ix_(~free_dof,~free_dof)],r_vec[~free_dof]) # 2N-2xfree points
                print('Incremental solution found')
                
                u_trial[~free_dof]=u_trial[~free_dof]+alpha*du # (2N,)
                #r_vec,history,phi[:,n],f_int[:,n]=getForce(x,u_trial,T,bn,family,partialAreas,surfaceCorrection,dof_vec,idb,ndof,bc_setn,V_DOF,par_omega,c,model,damage,history) # Update to include arbitrary displacement kinematic conditions
                r_vec,phi[:,n],f_int[:,n]=pd.general_functions.getForce(mesh,u_trial,bn,bc,family,bc_setn,V_DOF,par_omega,model,penalty) # Update to include arbitrary displacement kinematic conditions
                r=np.linalg.norm(r_vec[:ndof],np.inf)
                #r_max=epsilon*np.max([np.linalg.norm(bn*V,np.inf),np.linalg.norm(r_vec-bn*V,np.inf)])
                
                print(f'Iter: {_iter} | Residual equal to: {r} | np.maximum residual to be: {r_max}')
                _iter=_iter+1
            print(f'Solution found for the step {n+1} out of {n_tot}')
            un[:,n]=u_trial
            if n<n_tot-1:
                un[:,n+1]=u_trial # for the next iteration
        else:
            # Linear model
            u_trial=un[:,n]
            # Damage
            bc.damage.phi=phi[:,n] # Accessing current damage situation
            if n<2: # If the model is linear, there is no need to find the matrix more than once
                if model.stiffnessAnal==False:
                    #K=tangentStiffnessMatrix(mesh,u_trial,idb,family,partialAreas,surfaceCorrection,T,ndof,par_omega,c,model,damage,history)
                    K=pd.general_functions.tangentStiffnessMatrix(mesh,u_trial,family,bc,par_omega,model,penalty)
                else:
                    #K=analyticalStiffnessMatrix(mesh,u_trial,ndof,idb,family,partialAreas,surfaceCorrection,V,par_omega,c,model,damage,history)
                    K=model.analyticalStiffnessMatrix(mesh,u_trial,bc,family,par_omega,penalty)
            #bn=bn*V
            ff=bn*V_DOF
            if bc_set.size>0:
                # ff[ndof:]=penalty*bc_set[:,1]*((n+1)/n_tot)
                ff[ndof:]=penalty*bc_set[:,1]*((n+1)/n_tot)
            # du=np.linalg.lstsq(-K,ff,rcond=None)[0]
            du=np.linalg.solve(-K[np.ix_(~free_dof,~free_dof)],ff[~free_dof]) # 2N-2xfree points
            print(f'Solution found for the step {n+1} out of {n_tot}')
            un[~free_dof,n]=u_trial[~free_dof]+du
            #r_vec,history,phi[:,n],f_int[:,n]=getForce(x,un[:,n],T,bn,family,partialAreas,surfaceCorrection,dof_vec,idb,ndof,bc_set,V_DOF,par_omega,c,model,damage,history) # Update to include arbitrary displacement kinematic conditions
            r_vec,phi[:,n],f_int[:,n]=pd.general_functions.getForce(mesh,un[:,n],bn,bc,family,bc_set,V_DOF,par_omega,model,penalty) # Update to include arbitrary displacement kinematic conditions
            r=np.linalg.norm(r_vec[:ndof],np.inf)
        
        # Energy
        if model.dilatation==True:
            #theta,history.theta=dilatation(x,un[:,n],family,partialAreas,surfaceCorrection,[],idb,par_omega,c,model,damage,history,0)
            #theta,history.theta=dilatation(x,un[:,n],family,[],idb,par_omega,c,model,damage,history,0)
            theta=model.dilatation(mesh,un[:,n],family,bc,par_omega)
        if bc_set.size>0:
            con_dof=idb[bc_set[:,0].astype(int)]
        else:
            con_dof=[]
        for ii in range(0,len(x)):
            dofi=dof_vec[ii]
            if model.dilatation==True:
                #energy.W[ii,n]=strainEnergyDensity(x,un[:,n],theta,family[ii],partialAreas[ii],surfaceCorrection[ii],ii,idb,par_omega,c,model,damage,history.S[ii],history.theta)*V
                energy.W[ii,n]=model.strainEnergyDensity(mesh,un[:,n],theta,family,ii,bc,par_omega)*V
            else:
                #energy.W[ii,n]=strainEnergyDensity(x,un[:,n],[],family[ii],partialAreas[ii],surfaceCorrection[ii],ii,idb,par_omega,c,model,damage,history.S[ii])*V
                energy.W[ii,n]=model.strainEnergyDensity(mesh,un[:,n],family,ii,bc,par_omega)*V
            if n>0:
                bn_1=b*(n/n_tot) # b_(n-1)
                # External work realized by the displacement constraint
                if sum(dofi[0]==con_dof):
                    bn[dofi[0]]=-f_int[dofi[0],n]
                    bn_1[dofi[0]]=-f_int[dofi[0],n-1]
                    #du=u[con_dof,n]-u[con_dof,n-1]
                    #add_ext=bf*du
                if sum(dofi[1]==con_dof):
                    bn[dofi[1]]=-f_int[dofi[1],n]
                    bn_1[dofi[1]]=-f_int[dofi[1],n-1]
                energy.EW[ii,n]=np.dot(bn[dofi]+bn_1[dofi],un[dofi,n]-un[dofi,n-1]/2*V+energy.EW[ii,n-1])
            else:
                if sum(dofi[0]==con_dof):
                    bn[dofi[0]]=-f_int[dofi[0],n]
                if sum(dofi[1]==con_dof):
                    bn[dofi[1]]=-f_int[dofi[1],n]
                #energy.EW[ii,n]=np.dot(bn[dofi]-0,un[dofi,n]-0)/2*V
                energy.EW[ii,n]=np.dot(bn[dofi],un[dofi,n])/2*V
    
    return un,r,energy,phi

def solver_DynamicExplicit(mesh:pd.Mesh,family:pd.Family,bc:pd.BoundaryConditions,model:pd.Model,t:np.ndarray|list,rho:float|int,par_omega:tuple[float|int,int,int],data_dump:int=1,silent:bool=False):
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - family: family generetade with Family
    - bc: boundary conditions genretade with BoundaryConditions
    - model: model generated with Model
    - t: time array, ex: np.arange(0,t_final,dt)
    - rho: density [kg/m³]
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
    - data_dump: interval time steps to record outputs
    - silent: whether to print messages or not
    
    Output parameters:
    
    - t_s: t[n_sample-1]
    - u_n: displacements
    - phi: damage index
    - energy: 
        - W: macro-elastic energy
        - KE: kinect energy
        - EW: external work 
    - toc-tic: time to compute
    - F_load: load'''

    body_force=bc.bb

    dt=abs(t[1]-t[0])

    # Damage variables
    # Defining cracking trespassing matrix
    free_points=np.full((len(mesh.points)),False,dtype=bool)

    if bc.damage.brokenBonds is None:
        bc.initialize_brokenBonds(family)

    if bc.damage.crackIn is not None:
        crackSegments=len(bc.damage.crackIn) # At least 2

        # for ii in range(0,len(model.history_S)):
        for ii in range(0,len(mesh.points)):
            x_j=family.xj[ii]
                
            # brokenBonds=np.empty((len(x_j),crackSegments-1),dtype=bool)
            for kk in range(0,crackSegments-1):
                for jj in range(0,len(x_j)):
                    # brokenBonds[jj,kk]=checkBondCrack(x[ii],x_j[jj],bc.damage.crackIn[kk],bc.damage.crackIn[kk+1])[1]
                    bc.damage.brokenBonds[ii][jj]=pd.general_functions.checkBondCrack(mesh.points[ii],x_j[jj],bc.damage.crackIn[kk],bc.damage.crackIn[kk+1])[1]

            if np.all(bc.damage.brokenBonds[ii])==True:
                free_points[ii]=True
            
            # bc.damage.brokenBonds[ii,0:len(x_j),:]=brokenBonds # ii: node, jj: neighbors, kk: crack segment i to i+1
        if silent==False:
            print('Check for broken bonds done.')

    # Initialize simulation matrices
    Minv=1/rho # Diagonal and with the same value: scalar
    phi=np.zeros((len(mesh.points),len(t)))
    # Initial condition
    u_n=np.zeros((2*len(mesh.points),len(t))) # 2*N x n  2D matrix
    v_n=np.zeros((2*len(mesh.points),2)) # Velocity Verlet matrix [i i+1/2]: by initializing it to zero, the rigid motion is eliminated.ffz
    if body_force.ndim==1:
        # Constant body force
        bn=np.column_stack((body_force,body_force))
        flag_bf_constant=True
    else:
        if body_force.shape[1]!=len(t):
            bfnew=np.zeros((2*len(mesh.points),len(t)))      
            for iii in range(0,len(body_force)):
                f=interp1d(np.arange(0,body_force.shape[1]-1),body_force[iii])
                bfnew[iii]=f(np.arange(0,len(t))*(body_force.shape[1]-1)/len(t)-1)
            body_force=bfnew
        flag_bf_constant=False
    if data_dump>1:
        n_sample=np.hstack((1,np.arange(data_dump,len(t),data_dump)))
    else:
        n_sample=np.arange(data_dump,len(t),data_dump)
    if n_sample[-1]!=len(t):
        n_sample=np.hstack((n_sample,len(t)))
    t_s=t[n_sample-1]
    energy=pd.general_functions.Energy()
    energy.W=np.zeros((len(mesh.points),len(n_sample))) # No initial  deformation hence no initial strain energy density
    energy.KE=np.zeros((len(mesh.points),len(n_sample)))
    energy.EW=np.zeros((len(mesh.points),len(n_sample)))
    energy_ext_var=np.zeros((len(mesh.points))) # External energy that takes into account variable body force (velocity constraints)
    F_load=np.zeros((len(n_sample),2))
    fn=np.zeros((2*len(mesh.points))) # Initial force
    u_const=np.zeros((len(v_n)-bc.ndof)) # Constraint nodes
    # Temporary variables
    history=pd.general_functions.History()
    model.initialize_history(family)
    history.S=model.history.S
    if model.dilatation==True:
        history.T=model.history.theta
    else:
        history.T=np.array([])
    fn_temp=np.zeros_like(mesh.points)
    phi_temp=np.zeros((len(mesh.points)))

    # Recovering temporary files
    n_initial=0
    # if silent==False:
    #     print('Begining calculations.')
    tic=default_timer()
    for n in range(n_initial,len(t)-1):
        # Instatiate body force
        if flag_bf_constant==False:
            bn=body_force[:,n:n+2] # Increment b
        
        ############ VELOCITY VERLET ALGORITHM ###############
        # ---- Solving for the dof ----
        #### Step 1 - Midway velocity
        v_n[:bc.ndof,1]=v_n[:bc.ndof,0]+dt/2*Minv*(fn[:bc.ndof]+bn[:bc.ndof,0]) # V(n+1/2)
        #### Step 2 - Update displacement
        u_n[:bc.ndof,n+1]=u_n[:bc.ndof,n]+dt*v_n[:bc.ndof,1] # u_n(:,(2*(n+1)-1):2*(n+1)) = u_n(:,(2*n-1):2*n) + dt*v_n(:,3:4); % u(n+1)
        # ----- Solving for the displacement constraint nodes ----
        if u_const.size>0:
            if np.any(~np.isnan(bc.bc_set[:,1])) and np.any(~np.isnan(bc.bc_set[:,2])):
                raise UserWarning('The boundary condition matrix bc_set has prescribed both displacement and velocity for the same node')
            u_const[bc.bc_set[:,2]==0]=0 # Defining the displacements for the nodes with no velocity
            u_const=u_const+bc.bc_set[:,2]*dt # updatind the velocity constraint nodes
            u_n[bc.ndof:,n+1]=u_const
        u1=u_n[:,n]
        u2=u_n[:,n+1] # Vector of displacement that will be used to come back to 2D

        # Evaluating dilatation
        theta=np.zeros((len(mesh.points))) # Preallocate theta
        bc.damage.phi=phi[:,n] # Accessing current damage situation
        if model.dilatation==True:
            theta,history_T=model.dilatation('XXXXXXXXX')
            history.T=history_T
        
        ####### Step 3 - Update velocity
        #{Evaluate f[n+1]}
        fn=np.zeros((2*len(mesh.points))) # Instantiate force vector
        energy_pot=np.zeros(len(mesh.points)) # Pre-allocate potential energy
        energy_ext=np.zeros(len(mesh.points)) # Pre-allocate external force
        b_Weval=n+2%data_dump==0 or n+2==len(t) # Deciding when to evaluate the energy
        for ii in range(0,len(mesh.points)):
            fn_temp[ii],phi_temp[ii],energy_pot[ii]=pd.general_functions.parFor_loop(mesh,family,bc,model,u2,ii,par_omega,theta,b_Weval) # also updates model.history.S
        # Converting the temporary variables
        for ii in range(0,len(mesh.points)):
            fn[bc.dofi[ii]]=fn_temp[ii]
        phi[:,n+1]=phi_temp
        # Evaluate V(n+1)
        v_n[:bc.ndof,0]=v_n[:bc.ndof,1]+dt/2*Minv*(fn[:bc.ndof]+bn[:bc.ndof,0]) # V(n+1) is stored in the next V(n) using f n+1 and b n+1

        ## Evaluating energy
        if b_Weval==True:
            index_s=n_sample==n+2
            # Potential energy
            energy.W[:,index_s]=energy_pot.reshape((-1,1))
            # External work
            BBN=bn[:,0]
            if flag_bf_constant==True:
                # Constant body force
                # energy_ext=np.dot(u2[bc.dofi],BBN[bc.dofi])*mesh.A
                energy_ext=np.einsum('ij,ij->i',u2[bc.dofi],BBN[bc.dofi])*mesh.A
            else:
                du=u2[bc.dofi]-u1[bc.dofi]
                # energy_ext_var=energy_ext_var+np.dot(du,BBN[bc.dofi])*mesh.A
                energy_ext_var=energy_ext_var+np.einsum('ij,ij->i',u2[bc.dofi],BBN[bc.dofi])*mesh.A
            # External work realized by the velocity constraint
            if bc.bc_set.size>0:
                vel_dof=bc.idb[bc.bc_set[bc.bc_set[:,2]==0,0]].astype(int)
                v=bc.bc_set[bc.bc_set[:,2]==0,2]
            else:
                vel_dof=[]
                v=np.array([])
            bf=-fn[vel_dof]
            du=v*dt
            add_ext=bf*du
            for gg in range(0,len(vel_dof)):
                ind_vel=np.where(bc.dofi==vel_dof[gg])[0]
                ind_vel=ind_vel-(ind_vel>len(bc.dofi)-1)*len(bc.dofi)
                energy_ext_var[ind_vel]=energy_ext_var[ind_vel]+add_ext[gg]*mesh.A
            energy.EW[:,index_s]=(energy_ext+energy_ext_var).reshape((-1,1))
            # Kinectic energy
            for kk in range(0,len(mesh.points)):
                dofk=bc.dofi[kk]
                energy.KE[kk,index_s]=1/2*rho*np.linalg.norm(v_n[dofk,0])**2*mesh.A
            # Calculating load
            F_load[index_s,:]=pd.general_functions.fload(mesh,family,bc,model,u2,theta)
        else:
            # External incremental work only
            BBN=bn[:,0]
            if flag_bf_constant==False:
                du=u2[bc.dofi]-u1[bc.dofi]
                energy_ext_var=energy_ext_var+np.dot(du,BBN[bc.dofi])*mesh.A
            # External work realized by the velocity constraint
            if bc.bc_set.size>0:
                vel_dof=bc.idb[np.where(bc.bc_set[:,2]!=0)[0]]
                v=bc.bc_set[bc.bc_set[:,2]!=0,2]
            else:
                vel_dof=[]
                v=np.array([])
            bf=-fn[vel_dof]
            du=v*dt
            add_ext=bf*du
            for gg in range(0,len(vel_dof)):
                ind_vel=np.where(bc.dofi==vel_dof[gg])[0]
                ind_vel=ind_vel-(ind_vel>len(bc.dofi)-1)*len(bc.dofi)
                energy_ext_var[ind_vel]=energy_ext_var[ind_vel]+add_ext[gg]*mesh.A

        ############ COUNTING THE PROCESSING TIME #############
        toc=default_timer()
        if silent==False:
            # print(f'Time: {t[n]} s | Percentage: {(n+1)/(len(t)-1)*100} % | ETA: {(toc-tic)/(n+1)*(len(t)-(n+1))} s')
            pd.misc.progress_printer([(n+1)/(len(t)-1),t[n],(toc-tic)/(n+1)*(len(t)-(n+1))],fmt='Calculating... | Time: %v1 s | Percentage: |%b| %p % | ETA: %v2 s',bar_sep=40,hundred_percent_print=' Done.')

    # Sampling the results
    u_n=pd.general_functions.sampling(u_n,t,t_s)
    phi=pd.general_functions.sampling(phi,t,t_s)

    if silent==False:
        print('Sampling done.')

    return t_s,u_n,phi,energy,toc-tic,F_load
    