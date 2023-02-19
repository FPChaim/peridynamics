'''Provides some plotting functions'''

# from sys import float_info
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.cm import jet
from matplotlib.patches import Patch
import peridynamics as pd

def _3D_modification(mesh:pd.Mesh,u_2D:np.ndarray,bc:pd.BoundaryConditions,silent:bool=False) -> np.ndarray:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - u_2D: displacements (2D)
    - bc: boundary conditions genretade with BoundaryConditions
    - silent: whether to print messages or not
    
    Output parameters:
    
    - u: displacements (3D)'''
    
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

def plot_displacement(mesh:pd.Mesh,u:np.ndarray,bc:pd.BoundaryConditions,aspect:str='equalxy',scaleFactor:str|float|int='auto',return_fig_ax:bool=False,silent:bool=False,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64]) -> None:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - u: displacements
    - bc: boundary conditions genretade with BoundaryConditions
    - aspect:
        - "equalxy": x and y have the same aspect ratio (recommended)
        - "equal": x, y and z have the same aspect ratio
        - "auto": automatically chooses aspect ratios to fill image space with a "fuller" graphic (use if the others make the image difficult to see)
    - scaleFactor: scale used to increase the size of deformations, because usually they are very small and end up being indistinguible from the original configuration
        - "auto": let the algorithm decide
        - can also accept a number
    - return_fig_ax: if False (default), renders image. Else, returns figure and axis
    - silent: whether to print messages or not
    - fig_max_size_cm: maximum size in centimeters allowed for the picture before any aspect ratio rule kicks in
    
    Output parameters: 
    
    - rendered image or matplotlib figure and axis (if return_fig_ax=True)'''

    # Making a 3D matrix
    u=_3D_modification(mesh,u,bc,silent=silent)
    u=u[...,-1]

    # ϵ=float_info.epsilon
    ϵ=mesh.h/2

    x=mesh.points
    # Transforming into a matrix array
    # X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    X,Y=mesh.grid
    if len(u)==X.size: # rectangular mesh
        # sz=X.T.shape
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

    ax1.azim=-120
    ax2.azim=-120
    fig1.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax1.plot_surface(X=X,Y=Y,Z=V,cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Displacement Plot (x)',{'fontsize':24})
    ax1.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax1.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax1.set_zlabel(f'ux ({mesh.unit})',fontsize=18)
    ax1.set_aspect(aspect)

    ax2.plot_surface(X=X,Y=Y,Z=W,cmap='jet',rstride=1,cstride=1)
    ax2.set_title('Displacement Plot (y)',{'fontsize':24})
    ax2.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax2.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax2.set_zlabel(f'uy ({mesh.unit})',fontsize=18)
    ax2.set_aspect(aspect)

    if scaleFactor=='auto':
        scaleFactor=1/5*np.abs(np.max([np.max(x[:,0])-np.min(x[:,0]),np.max(x[:,1])-np.min(x[:,1])])/np.max(u,axis=None))

    if aspect=='equalxy': # correcting for 2D plots
        aspect=1

    fig2,ax3=plt.subplots()

    fig2.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax3.plot(X,Y,linestyle='-',color='blue')
    ax3.plot(X.T,Y.T,linestyle='-',color='blue')
    ax3.plot(X+scaleFactor*V,Y+scaleFactor*W,linestyle='-',color='green')
    ax3.plot((X+scaleFactor*V).T,(Y+scaleFactor*W).T,linestyle='-',color='green')
    legend_elements=[Patch(facecolor='none',edgecolor='blue',label='Reference'),
                     Patch(facecolor='none',edgecolor='green',label='Deformed')]
    ax3.legend(handles=legend_elements)
    ax3.set_title(f'Mesh Displacement (Scale Factor = {scaleFactor:.2e})',{'fontsize':24})
    ax3.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax3.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax3.set_aspect(aspect)

    fig3,ax4=plt.subplots()

    fig3.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax4.scatter(mesh.x,mesh.y,marker='o',edgecolors='blue',facecolors='none')
    ax4.quiver(mesh.x,mesh.y,u[:,0],u[:,1],angles='xy')
    ax4.set_title('Displacement Quiver',{'fontsize':24})
    ax4.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax4.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax4.set_aspect(aspect)

    fig4,ax4=plt.subplots()

    fig4.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax4.scatter(mesh.x,mesh.y,marker='o',edgecolors='blue',facecolors='blue',label='Reference')
    ax4.scatter(mesh.x+u[:,0]*scaleFactor,mesh.y+u[:,1]*scaleFactor,marker='o',edgecolors='green',facecolors='green',label='Deformed')
    ax4.legend()
    ax4.set_title(f'Mesh Displacement Scatter (Scale Factor = {scaleFactor:.2e})',{'fontsize':24})
    ax4.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax4.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax4.set_aspect(aspect)

    if return_fig_ax==True:
        figs=[fig1,fig2,fig3,fig4]
        axs=[ax1,ax2,ax3,ax4]
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        return figs,axs

    plt.show()

def plot_strain(mesh:pd.Mesh,u:np.ndarray,bc:pd.BoundaryConditions,aspect:str='equalxy',return_fig_ax:bool=False,silent:bool=False,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64]) -> None:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - u: displacements
    - bc: boundary conditions genretade with BoundaryConditions
    - aspect:
        - "equalxy": x and y have the same aspect ratio (recommended)
        - "equal": x, y and z have the same aspect ratio
        - "auto": automatically chooses aspect ratios to fill image space with a "fuller" graphic (use if the others make the image difficult to see)
    - return_fig_ax: if False (default), renders image. Else, returns figure and axis
    - silent: whether to print messages or not
    - fig_max_size_cm: maximum size in centimeters allowed for the picture before any aspect ratio rule kicks in
    
    Output parameters: 
    
    - rendered image or matplotlib figure and axis (if return_fig_ax=True)'''

    # Making a 3D matrix
    u=_3D_modification(mesh,u,bc,silent=silent)
    u=u[...,-1]

    # ϵ=float_info.epsilon # very small number of the order of the machine precision
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
    N=mesh.ncols # number of nodes in a row
    for i in range(0,len(mesh.points)):
        if ~np.any(out_layers==i):
            # Bulk nodes
            exx[i]=(v[i+1]-v[i-1])/2/mesh.h # dv/dx
            eyy[i]=(w[i+N]-w[i-N])/2/mesh.h # dw/dy
            exy[i]=1/4/mesh.h*(v[i+N]-v[i-N]+w[i+1]-w[i-1]) # 1/2*(dv/dy + dw/dx)
    # Transforming into matrix
    # X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    X,Y=mesh.grid
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

    ax1.azim=-120
    ax2.azim=-120
    ax3.azim=-120
    fig1.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax1.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EXX[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Strain Plot (x)',{'fontsize':24})
    ax1.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax1.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax1.set_zlabel('$\epsilon_x$',fontsize=18)
    ax1.set_aspect(aspect)

    ax2.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EYY[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax2.set_title('Strain Plot (y)',{'fontsize':24})
    ax2.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax2.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax2.set_zlabel('$\epsilon_y$',fontsize=18)
    ax2.set_aspect(aspect)

    ax3.plot_surface(X=X[1:-1,1:-1],Y=Y[1:-1,1:-1],Z=EXY[1:-1,1:-1],cmap='jet',rstride=1,cstride=1)
    ax3.set_title('Strain Plot (xy)',{'fontsize':24})
    ax3.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax3.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax3.set_zlabel('$\epsilon_{xy}$',fontsize=18)
    ax3.set_aspect(aspect)

    if return_fig_ax==True:
        fig=fig1
        axs=[ax1,ax2,ax3]
        plt.close()
        return fig,axs
        
    plt.show()

def plot_damage(mesh:pd.Mesh,phi:np.ndarray,method:str='pcolormesh',return_fig_ax:bool=False,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64]) -> None:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - phi: damage index
    - method: 
        - "pcolormesh": default matplotlib meshing method (recommended)
        - "pcolor": slower, use if pcolormesh fails
    - return_fig_ax: if False (default), renders image. Else, returns figure and axis
    - fig_max_size_cm: maximum size in centimeters allowed for the picture before any aspect ratio rule kicks in
    
    Output parameters: 
    
    - rendered image or matplotlib figure and axis (if return_fig_ax=True)'''

    # ϵ=float_info.epsilon # very small number of the order of the machine precision
    ϵ=mesh.h/2

    # X,Y=np.meshgrid(np.arange(mesh.x.min(),mesh.x.max()+ϵ,mesh.h),np.arange(mesh.y.min(),mesh.y.max()+ϵ,mesh.h))
    X,Y=mesh.grid
    # PHI=np.zeros(X.shape)

    # sz=X.T.shape
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

    fig.set_size_inches(np.array(fig_max_size_cm)/2.54)

    if method=='pcolormesh':
        cax=ax.pcolormesh(X,Y,PHI,cmap='jet')#,vmin=0,vmax=1,norm='linear'
    elif method=='pcolor':
        cax=ax.pcolor(X,Y,PHI,cmap='jet')
    else:
        raise NameError('Expected methods are: "pcolormesh" and "pcolor"')
    ax.set_title('Damage index',{'fontsize':24})
    ax.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax.set_aspect(1)
    fig.colorbar(cax)
    
    if return_fig_ax==True:
        plt.close()
        return fig,ax
        
    plt.show()

def plot_crack(mesh:pd.Mesh,phi:np.ndarray,dt,data_dump:int=80,return_fig_ax:bool=False,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64]) -> None:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - phi: damage index
    - dt: time interval
    - data_dump: timesteps to dump data
    - return_fig_ax: if False (default), renders image. Else, returns figure and axis
    - fig_max_size_cm: maximum size in centimeters allowed for the picture before any aspect ratio rule kicks in
    
    Output parameters: 
    
    - rendered image or matplotlib figure and axis (if return_fig_ax=True)'''

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

    fig.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax.plot(np.arange(0,samples+1)*(data_dump*dt),V_l)
    ax.set_title('Velocity plot',{'fontsize':24})
    ax.set_xlabel(f'Simulation time',fontsize=18)
    ax.set_ylabel(f'Velocity of the crack tip ({mesh.unit}/s)',fontsize=18)
    ax.set_aspect(1)

    if return_fig_ax==True:
        plt.close()
        return fig,ax
        
    plt.show()

def plot_energy(mesh:pd.Mesh,energy:pd.general_functions.Energy,aspect:str='equalxy',return_fig_ax:bool=False,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64],legend_loc:str='lower right',legend_bbox_to_anchor:tuple[float|int,float|int,float|int,float|int]=(1,1,0,0),legend_framealpha:float|int=1,legend_edgecolor:str='grey') -> None:
    '''Input parameters:
    
    - mesh: peridynamic mesh generated with Mesh
    - energy - energy object with the following attributes:
        - W: macro-elastic energy
        - KE: kinect energy
        - EW: external work
    - aspect:
        - "equalxy": x and y have the same aspect ratio (recommended)
        - "equal": x, y and z have the same aspect ratio
        - "auto": automatically chooses aspect ratios to fill image space with a "fuller" graphic (use if the others make the image difficult to see)
    - return_fig_ax: if False (default), renders image. Else, returns figure and axis
    - fig_max_size_cm: maximum size in centimeters allowed for the picture before any aspect ratio rule kicks in
    - legend_loc, legend_bbox_to_anchor, legend_framealpha, legend_edgecolor: legend parameters
    
    Output parameters: 
    
    - rendered image or matplotlib figure and axis (if return_fig_ax=True)'''

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

    ax1.azim=-120
    fig1.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax1.plot_surface(X=X,Y=Y,Z=WW,cmap='jet',rstride=1,cstride=1)
    ax1.set_title('Strain Energy Density',{'fontsize':24})
    ax1.set_xlabel(f'x ({mesh.unit})',fontsize=18)
    ax1.set_ylabel(f'y ({mesh.unit})',fontsize=18)
    ax1.set_zlabel(f'Energy (J/{mesh.unit})',fontsize=18)
    ax1.set_aspect(aspect)

    # Total energy convolution
    t=np.arange(1,n_final+1) # number of time steps
    fig2,ax2=plt.subplots()

    fig2.set_size_inches(np.array(fig_max_size_cm)/2.54)

    ax2.plot(t,np.sum(energy.W,axis=0),label='Strain energy')
    ax2.plot(t,np.sum(energy.KE,axis=0),label='Kinect energy')
    ax2.plot(t,np.sum(energy.EW,axis=0),label='External work')
    ax2.plot(t,np.sum(energy.W+energy.KE-energy.EW,axis=0),label='Total internal energy') # Total energy for each point
    ax2.set_title('Energy Revolution',{'fontsize':24})
    ax2.set_xlabel(f'Time step',fontsize=18)
    ax2.set_ylabel(f'Energy (J/{mesh.unit})',fontsize=18)
    ax2.legend(loc=legend_loc,bbox_to_anchor=legend_bbox_to_anchor,framealpha=legend_framealpha,edgecolor=legend_edgecolor)

    if return_fig_ax==True:
        figs=[fig1,fig2]
        axs=[ax1,ax2]
        plt.close(fig1)
        plt.close(fig2)
        return figs,axs
        
    plt.show()
    