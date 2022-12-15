# from sys import float_info
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.cm import jet
from matplotlib.patches import Patch

def _3D_modification(mesh,u_2D,bc,silent=False):
    '''Input parameters:
    
    - mesh: 
    - u_2D: 
    - bc: 
    - silent: 
    
    Output parameters:
    
    -u: '''
    
    x=mesh.points
    # idb=bc.idb
    dofi=bc.dofi

    if len(u_2D)==2*len(x):
        u=np.zeros((len(x),2,u_2D.shape[1]))
        # ii=np.arange(0,len(x))
        # dofi=np.hstack((idb[2*ii].reshape((-1,1)),idb[2*ii+1].reshape((-1,1)))) # Nx2
        for n in range(0,u_2D.shape[1]):
            # u_temp=u_2D[:,n]
            # u[...,n]=u_temp[dofi]
            u[...,n]=u_2D[dofi,n]
    else:
        u=u_2D
    
    if silent==False:
        print('Displacement vector converted.')

    return u

def plot_displacement(mesh,u,bc,aspect='equalxy',scaleFactor='auto',return_fig_ax=False,silent=False):
    '''Input parameters:
    
    - mesh: 
    - u: 
    - aspect:
        - "equalxy": 
        - "equal": 
        - "auto": 
    - scaleFactor: 
    
    Output parameters: '''

    # Making a 3D matrix
    u=_3D_modification(mesh,u,bc,silent=silent)
    u=u[...,-1]

    #ϵ=float_info.epsilon
    ϵ=mesh.h/2

    x=mesh.points
    #b=max(x[:,0])-min(x[:,0])
    #a=max(x[:,1])-min(x[:,1])
    # Transforming into a matrix array
    #X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    X,Y=mesh.grid
    if len(u)==X.size: # rectangular mesh
        #sz=X.T.shape
        sz=X.shape
        V=u[:,0].reshape(sz)
        W=u[:,1].reshape(sz)
    else: # non rectangular mesh
        V=np.zeros_like(X)
        W=np.zeros_like(X)
        for jjj in range(0,X.shape[1]):
            for iii in range(0,Y.shape[0]):
                coord=[X[0,jjj],Y[iii,0]]
                ind=None
                for ii in range(0,len(x)):
                    if x[ii,0]<coord[0]+ϵ and x[ii,0]>coord[0]-ϵ and x[ii,1]<coord[1]+ϵ and x[ii,1]>coord[1]-ϵ:
                        ind=ii
                        break
                if ind is not None:
                    V[iii,jjj]=u[ind,0]
                    W[iii,jjj]=u[ind,1]
                else:
                    V[iii,jjj]=np.nan
                    W[iii,jjj]=np.nan
        
    fig1,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,subplot_kw=dict(projection='3d'))
    ax1.plot_surface(X=X,Y=Y,Z=V,cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Displacement Plot (x)')
    ax1.set_xlabel(f'x ({mesh.unit})')
    ax1.set_ylabel(f'y ({mesh.unit})')
    ax1.set_zlabel(f'ux ({mesh.unit})')
    ax1.set_aspect(aspect)

    ax2.plot_surface(X=X,Y=Y,Z=W,cmap='jet',rstride=1,cstride=1)
    ax2.set_title('Displacement Plot (y)')
    ax2.set_xlabel(f'x ({mesh.unit})')
    ax2.set_xlabel(f'y ({mesh.unit})')
    ax2.set_zlabel(f'uy ({mesh.unit})')
    ax2.set_aspect(aspect)

    if scaleFactor=='auto':
        scaleFactor=1/5*np.abs(np.max([np.max(x[:,0])-np.min(x[:,0]),np.max(x[:,1])-np.min(x[:,1])])/np.max(u,axis=None))

    if aspect=='equalxy': # correcting for 2D plots
        aspect=1

    fig2,ax3=plt.subplots()
    ax3.plot(X,Y,linestyle='-',color='blue')
    ax3.plot(X.T,Y.T,linestyle='-',color='blue')
    ax3.plot(X+scaleFactor*V,Y+scaleFactor*W,linestyle='-',color='green')
    ax3.plot((X+scaleFactor*V).T,(Y+scaleFactor*W).T,linestyle='-',color='green')
    legend_elements=[Patch(facecolor='none',edgecolor='blue',label='Reference'),
                     Patch(facecolor='none',edgecolor='green',label='Deformed')]
    ax3.legend(handles=legend_elements)
    ax3.set_title(f'Mesh Displacement (Scale Factor = {scaleFactor:.2e})')
    ax3.set_xlabel(f'x ({mesh.unit})')
    ax3.set_xlabel(f'y ({mesh.unit})')
    ax3.set_aspect(aspect)

    fig3,ax4=plt.subplots()
    ax4.scatter(mesh.x,mesh.y,marker='o',edgecolors='blue',facecolors='none')
    ax4.quiver(mesh.x,mesh.y,u[:,0],u[:,1],angles='xy')
    ax4.set_title('Displacement Quiver')
    ax4.set_xlabel(f'x ({mesh.unit})')
    ax4.set_xlabel(f'y ({mesh.unit})')
    ax4.set_aspect(aspect)

    fig4,ax4=plt.subplots()
    ax4.scatter(mesh.x,mesh.y,marker='o',edgecolors='blue',facecolors='blue',label='Reference')
    ax4.scatter(mesh.x+u[:,0]*scaleFactor,mesh.y+u[:,1]*scaleFactor,marker='o',edgecolors='green',facecolors='green',label='Deformed')
    ax4.legend()
    ax4.set_title(f'Mesh Displacement Scatter (Scale Factor = {scaleFactor:.2e})')
    ax4.set_xlabel(f'x ({mesh.unit})')
    ax4.set_xlabel(f'y ({mesh.unit})')
    ax4.set_aspect(aspect)

    if return_fig_ax==True:
        figs=[fig1,fig2,fig3,fig4]
        axs=[ax1,ax2,ax3,ax4]
        plt.close()
        return figs,axs

    plt.show()

def plot_strain(mesh,u,bc,aspect='equalxy',return_fig_ax=False,silent=False):
    '''Input parameters:
    
    - mesh: 
    - u: 
    - aspect:
        - "equalxy": 
        - "equal": 
        - "auto": 
    - silent:
    
    Output parameters: '''

    # Making a 3D matrix
    u=_3D_modification(mesh,u,bc,silent=silent)
    u=u[...,-1]

    #ϵ=float_info.epsilon # very small number of the order of the machine precision
    ϵ=mesh.h/2

    exx=np.zeros(len(mesh.points))
    eyy=np.zeros(len(mesh.points))
    exy=np.zeros(len(mesh.points))

    v=u[:,0] # x component of the displacement field
    w=u[:,1] # y component of the displacement field
    x_lim=[mesh.x.min(),mesh.x.max()]
    y_lim=[mesh.y.min(),mesh.y.max()]
    bottom=np.where(mesh.y<(y_lim[0]+ϵ))[0]
    top=np.where(mesh.y>(y_lim[1]-ϵ))[0]
    left=np.where(mesh.x<(x_lim[0]+ϵ))[0]
    right=np.where(mesh.x>(x_lim[1]-ϵ))[0]
    out_layers=np.hstack((top,bottom,left,right))
    # N=0 # number of nodes in a row
    # for i in range(0,len(mesh.points)):
    #     if mesh.points[i,1]!=mesh.points[0,1]:
    #         break
    #     N=N+1
    N=mesh.ncols # number of nodes in a row
    for i in range(0,len(mesh.points)):
        if ~np.any(out_layers==i):
            # Bulk nodes
            exx[i]=(v[i+1]-v[i-1])/2/mesh.h # dv/dx
            eyy[i]=(w[i+N]-w[i-N])/2/mesh.h # dw/dy
            exy[i]=1/4/mesh.h*(v[i+N]-v[i-N]+w[i+1]-w[i-1]) # 1/2*(dv/dy + dw/dx)
    # Transforming into matrix
    X,Y=mesh.grid
    # X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    EXX=np.zeros(X.shape)
    EYY=np.zeros(Y.shape)
    EXY=np.zeros(X.shape)
    for j in range(0,X.shape[1]):
        for i in range(0,Y.shape[0]):
            coord=[X[0,j],Y[i,0]]
            ind=None
            for ii in range(0,len(mesh.points)):
                if mesh.points[ii,0]<coord[0]+ϵ and mesh.points[ii,0]>coord[0]-ϵ\
                and mesh.points[ii,1]<coord[1]+ϵ and mesh.points[ii,1]>coord[1]-ϵ:
                    ind=ii
                    break
            if ind is not None:
                EXX[i,j]=exx[ind]
                EYY[i,j]=eyy[ind]
                EXY[i,j]=exy[ind]
            else:
                EXX[i,j]=np.nan
                EYY[i,j]=np.nan
                EXY[i,j]=np.nan
    # Plot
    #fig1,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,subplot_kw=dict(projection='3d'))
    fig1,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3,subplot_kw=dict(projection='3d'))
    ax1.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EXX[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Strain Plot (x)')
    ax1.set_xlabel(f'x ({mesh.unit})')
    ax1.set_xlabel(f'y ({mesh.unit})')
    ax1.set_zlabel('$\epsilon_x$')
    ax1.set_aspect(aspect)

    ax2.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EYY[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax2.set_title('Strain Plot (y)')
    ax2.set_xlabel(f'x ({mesh.unit})')
    ax2.set_xlabel(f'y ({mesh.unit})')
    ax2.set_zlabel('$\epsilon_y$')
    ax2.set_aspect(aspect)

    ax3.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EXY[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax3.set_title('Strain Plot (xy)')
    ax3.set_xlabel(f'x ({mesh.unit})')
    ax3.set_xlabel(f'y ({mesh.unit})')
    ax3.set_zlabel('$\epsilon_{xy}$')
    ax3.set_aspect(aspect)

    if return_fig_ax==True:
        fig=fig1
        axs=[ax1,ax2,ax3]
        plt.close()
        return fig,axs
        
    plt.show()

def plot_damage(mesh,phi,method='pcolormesh',return_fig_ax=False):
    '''Input parameters:
    
    - mesh: 
    - phi: 
    - method: 
        - "pcolormesh": 
        - "pcolor": 
    
    Output parameters: '''

    #ϵ=float_info.epsilon # very small number of the order of the machine precision
    ϵ=mesh.h/2

    #X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    X,Y=mesh.grid
    # PHI=np.zeros(X.shape)

    #sz=X.T.shape
    sz=X.shape

    for n in range(0,phi.shape[1]):
        if len(mesh.points)==X.size: # rectangular mesh
            PHI=phi[:,n].reshape(sz)#.T
        else:
            phi_temp=np.zeros((X.size))
            # X=X.T
            # Y=Y.T
            for ii in range(0,X.size):
                index=np.where(np.logical_and(np.logical_and(mesh.x<X[ii]+ϵ,mesh.x>X[ii]-ϵ),np.logical_and(mesh.y<Y[ii]+ϵ,mesh.y>Y[ii]-ϵ)))[0]
                if index.size>0:
                    phi_temp[ii]=phi[index,n]
                else:
                    phi_temp[ii]=np.nan
            # X=X.T
            # Y=Y.T
            PHI=phi_temp.reshape(sz)

    # for j in range(0,X.shape[1]):
    #     for i in range(0,Y.shape[0]):
    #         coord=[X[0,j],Y[i,0]]
    #         ind=None
    #         for ii in range(0,len(mesh.points)):
    #             if mesh.points[ii,0]<coord[0]+ϵ and mesh.points[ii,0]>coord[0]-ϵ\
    #             and mesh.points[ii,1]<coord[1]+ϵ and mesh.points[ii,1]>coord[1]-ϵ:
    #                 ind=ii
    #                 break
    #         if ind is not None:
    #             PHI[i,j]=phi[ind]
    #             #PHI[i,j]=np.random.randn() # test
    #         else:
    #             PHI[i,j]=np.nan

    fig,ax=plt.subplots()
    if method=='pcolormesh':
        cax=ax.pcolormesh(X,Y,PHI,cmap='jet')#,vmin=0,vmax=1,norm='linear'
    elif method=='pcolor':
        cax=ax.pcolor(X,Y,PHI,cmap='jet')
    else:
        raise NameError('Expected methods are: "pcolormesh" and "pcolor"')
    ax.set_title('Damage index')
    ax.set_xlabel(f'x ({mesh.unit})')
    ax.set_xlabel(f'y ({mesh.unit})')
    ax.set_aspect(1)
    fig.colorbar(cax)
    
    if return_fig_ax==True:
        plt.close()
        return fig,ax
        
    plt.show()

def plot_crack(mesh,phi,dt,data_dump=80,return_fig_ax=False):
    '''Input parameters:
    
    - mesh: 
    - phi: 
    - n_final: 
    - dt: 
    - data_dump: 
    
    Output parameters: '''

    # data_dump=80 # Every 80 timesteps
    n_final=phi.shape[1]
    if n_final<data_dump:
        data_dump=n_final
    samples=np.floor(n_final/data_dump).astype(int)
    V_l=np.zeros(samples)
    for n_samp in range(0,samples):
        n=data_dump*n_samp
        set_dam=np.where(phi[:,n]>0.35)[0] # Binary of greater than 0.35 damage index
        x_dam=mesh.points[set_dam] # Set of probable tips
        x_max=x_dam[:,0].max()
        if set_dam.size>0:
            damx_ind=(x_dam[:,0]==x_max)
            set_dam=set_dam[damx_ind]
            if len(set_dam)>1:
                y_max=np.max(x_dam[damx_ind,1])
                damy_ind=(x_dam[damx_ind,1])>y_max-1e-12
                set_dam=set_dam[damy_ind]
                # tip=np.zeros((len(set_dam),2))
                tip=np.array([0.,0.])
                tip[1]=set_dam
            else:
                tip=np.array([0.,0.])
                # tip=np.zeros((len(set_dam),2))
                tip[1]=set_dam
            if n_samp>0:
                V_l[n_samp+1]=np.linalg.norm(mesh.points[tip[1],0]-mesh.points[tip[0],0])/dt/data_dump
                tip[0]=tip[1]
            else:
                tip[0]=set_dam
    # Plot the velocity
    fig,ax=plt.subplots()
    ax.plot(np.arange(0,samples+1)*(data_dump*dt),V_l)
    ax.set_title('Velocity plot')
    ax.set_xlabel(f'Simulation time')
    ax.set_xlabel(f'Velocity of the crack tip ({mesh.unit}/s)')
    ax.set_aspect(1)

    if return_fig_ax==True:
        plt.close()
        return fig,ax
        

    plt.show()

def plot_energy(mesh,energy,aspect='equalxy',return_fig_ax=False):

    ϵ=mesh.h/2
    n_final=energy.W.shape[1]

    # Strain energy density
    X,Y=mesh.grid
    WW=np.zeros_like(X)
    for j in range(0,X.shape[1]):
        for i in range(0,Y.shape[0]):
            coord=[X[0,j],Y[i,0]]
            ind=None
            for ii in range(0,len(mesh.points)):
                if mesh.points[ii,0]<coord[0]+ϵ and mesh.points[ii,0]>coord[0]-ϵ\
                and mesh.points[ii,1]<coord[1]+ϵ and mesh.points[ii,1]>coord[1]-ϵ:
                    ind=ii
                    break
            if ind is not None:
                WW[i,j]=energy.W[ind,n_final-1]
            else:
                WW[i,j]=np.nan

    fig1,ax1=plt.subplots(nrows=1,ncols=1,subplot_kw=dict(projection='3d'))
    ax1.plot_surface(X=X,Y=Y,Z=WW,cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Strain Energy Density')
    ax1.set_xlabel(f'x ({mesh.unit})')
    ax1.set_xlabel(f'y ({mesh.unit})')
    ax1.set_zlabel('W (J/m)')
    ax1.set_aspect(aspect)

    # Total energy convolution
    t=np.arange(1,n_final+1) # number of time steps
    fig2,ax2=plt.subplots()
    ax2.plot(t,np.sum(energy.W,axis=0),label='Strains energy')
    ax2.plot(t,np.sum(energy.KE,axis=0),label='Kinect energy')
    ax2.plot(t,np.sum(energy.EW,axis=0),label='External work')
    ax2.plot(t,np.sum(energy.W+energy.KE-energy.EW,axis=0),label='Total internal energy') # Total energy for each point
    ax2.set_title('Strain Energy Density')
    ax2.set_xlabel(f'Time step')
    ax2.set_xlabel(f'Energy (')
    ax2.legend()
    # if aspect=='equalxy':
    #     aspect=1
    # ax2.set_aspect(aspect)

    if return_fig_ax==True:
        figs=[fig1,fig2]
        axs=[ax1,ax2]
        plt.close()
        return figs,axs
        
    plt.show()

# def PostProcessing(mesh,u,n,bc,energy,phi,dt=0,silent=False):
#     '''Input parameters:
    
#     - mesh: 
#     - u: <-- u_n from solver_QuasiStatic
#     - n: number of load steps <-- n_tot from solver_QuasiStatic
#     - bc: 
#     - energy: 
#     - phi: 
#     - dt:
#     - silent: 
    
#     Output parameters: '''

#     #x=mesh.points
#     #idb=bc.idb


#     if u.size>0:
#         # Making a 3D matrix
#         u=_3D_modification(mesh,u,bc,silent=silent)
#         # Plot the displacement and strain map
#         displacementPlot(mesh,u[...,n-1],scaleFactor='auto')
#         strainPlot(mesh,u[...,n-1],silent=silent) # To be perfected

#     # Plot the damage index
#     if phi.size>0:
#         damagePlot(mesh,phi[:,n-1])
#         # Plot the crack properties
#         # if dt>0:
#         #     trackCrack(mesh,phi,n,dt)
    
