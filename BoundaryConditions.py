'''Provides a way to set 2D peridynamics boundary conditions'''

from functools import cached_property as _cached_property
import numpy as np
import matplotlib.pyplot as plt
import peridynamics as pd

class BoundaryConditions:
    '''A class that one should intializate as an object, e.g: bc=BoundaryConditions() and use one of the following methods to get started:
    - set_body_forces
    - set_displacement_constraints
    - set_velocities_constraints

    It's also possible to manually interact with the family by utilizing the following methods for debugging purposes, though the solvers do that automatically:
    - compute_dofj(family)
    - compute_noFail_ij(family)
    - initialize_brokenBonds(family)
    
    Input:
    
    - Varies depending on the method used. See each method for more info
    
    Output:

    Updates the following class parameters:
    
    - b: Nx2 array of body force
    - bb: 2N array of body force corrected with the set constraints
    - bc_set: contains the set of constrained degrees of freedom on the first collumn and its corresponding value on the second collumn; the third collumn contains the corresponding dof velocity
        2Nx3 array with the following structure:
        [Node index (2N), Displacement constraint value (2N), Velocity constraint value (2N)]
    - bodyForce (need to utilize the method set_body_forces): structured array with the following configuration:
        [Node index, Force x value, Force y value]
        Where:
        - Node index: nodes where the stresses where applied | type: array of ints
        - Force x value: values of the force in the x direction | type: arry of floats
        - Force y value: values of the force in the y direction | type: arry of floats
    - damage - damage object with the following structure:
        - alpha: damage parameter
        - beta: damage parameter
        - brokenBonds (only if the methdod initialize_brokenBonds(family) was used): brokenBonds vector (initialization only) | ii: node, jj: neighbors, kk: crack segment i to i+1
        - crackIn: Nx2 np.array of initial crack segments, where N≥2
        - damage_dependent_Sc: True if Sc is damage dependent
        - damage_on: True if applying damage to the model
        - gamma: damage parameter
        - phi: damage index (actually not touched here. Calculated in the solver)
        - Sc: critical relative elongation
        - thetaC: dilatation parameter ?
    - disp (need to utilize the method set_displacement_constraints): structured array of displacement constraints. See the method set_displacement_constraints for more info
    - dofi: IDB vector in Nx2 condition
    - dofj (only if the methdod compute_dofj(family) was used): 
    - idb: collumn vector that index each of its row (corresponding to the actual dof) to the matrix dof. Example: idb = [1 3 4 6 7 8] means that the third row of the stiffness matrix is related to the 2nd dof freedom of the system
    - ndof: number of the mesh degree of freedoms
    - noFail: set of nodes for which we have no fail condition (mu = 1 always)
    - noFail_ij (only if the methdod compute_noFail_ij(family) was used): 
    - sigma_x: stresses first collumn
    - sigma_y: stresses second collumn
    - tau_xy: stresses third collumn
    - vel (need to utilize the method set_velocities_constraints): structured array of displacement velocities. See the method set_displacement_constraints for more info'''
    

    def __init__(self) -> None:
        
        #from numpy import np.array
        self.bodyForce=np.array([])
        self.b=np.array([])
        # self.bb=np.array([])
        self.disp=np.array([])
        self.vel=np.array([])
        self.noFail=np.array([])
        self._bc_set=np.empty((0,3))
        self.damage=pd.general_functions.Damage()

    def set_body_forces(self,mesh:pd.Mesh,stresses:np.ndarray|list,stress_nodes:np.ndarray|list) -> None:
        '''Input:
        
        - mesh: peridynamic mesh generated with Mesh
        - stresses: applied normal and shear stresses - [σx,σy,τxy] N/m² | type: np.array or list
            - Can habe the shape (3,) --> for constant stresses
            - Can have the shape (3,N) --> for N variations of stresses
        - stress_nodes: which nodes are being stressed | type: np.array or list of ints
        
        Output:

        Updates the following class attributes:

        - b: Nx2 array of body force
        - bb: 2N array of body force corrected with the set constraints
        - bodyForce (need to utilize the method set_body_forces): structured array with the following configuration:
            [Node index, Force x value, Force y value]
            Where:
            - Node index: nodes where the stresses where applied | type: array of ints
            - Force x value: values of the force in the x direction | type: arry of floats
            - Force y value: values of the force in the y direction | type: arry of floats
        - sigma_x: stresses first collumn
        - sigma_y: stresses second collumn
        - tau_xy: stresses third collumn'''

        self._mesh=mesh

        stresses=np.atleast_2d(stresses)

        self.sigma_x=stresses[:,0] # σx
        self.sigma_y=stresses[:,1] # σy
        self.tau_xy=stresses[:,2] # τxy
        
        # Array structure
        self.bodyForce=np.empty_like(stress_nodes,
            dtype=[('Node index',int),('Force x value',float),('Force y value',float)])
        self.bodyForce['Node index']=stress_nodes
        self.bodyForce['Force x value']=self.sigma_x
        self.bodyForce['Force y value']=self.sigma_y

        # (N,2)
        self.b=np.zeros_like(mesh.points,dtype=float)
        self.b[stress_nodes,0]=self.sigma_x
        self.b[stress_nodes,1]=self.sigma_y

        # bb --> (2N,)

        self._update_bc_set('bodyforces') # only initializes it
        self._update_noFail('bodyforces')
        
    def set_displacement_constraints(self,mesh:pd.Mesh,
    disp_nodes:np.ndarray|list,disp_bools_x:np.ndarray|list,disp_bools_y:np.ndarray|list,disp_values_x:np.ndarray|list,disp_values_y:np.ndarray|list) -> None:
        '''Input:

        - mesh: peridynamic mesh generated with Mesh
        - disp_nodes: which nodes have constrained displacements | type np.array or list of ints
        - disp_bools_x: wheter the displacement in the x direction is prescribed | type: np.array or list of bools
        - disp_bools_y: wheter the displacement in the y direction is prescribed | type: np.array or list of bools
        - disp_values_x: values of prescribed displacement in the x direction | type: np.array or list of ints
        - disp_values_y: values of prescribed displacement in the y direction | type: np.array or list of ints

        Output:

        Updates class attribute vel with a structured np.array:
        [Node index, Displacement x is prescribed, Displacement y is prescribed, Displacement x value, Displacement y value]
        
        Where:

        - Node index: which nodes have constrained displacements | type: np.array of ints
        - Displacement x is prescribed: wheter the displacement in the x direction is prescribed | type: np.array of bools
        - Displacement y is prescribed: wheter the displacement in the y direction is prescribed | type: np.array of bools
        - Displacement x value: values of prescribed displacement in the x direction | type: np.array of ints
        - Displacement y value: values of prescribed displacement in the y direction | type: np.array of ints'''

        self._mesh=mesh

        self.disp=np.empty_like(disp_nodes,
            dtype=[('Node index',int),('Displacement x is prescribed',bool),('Displacement y is prescribed',bool),
            ('Displacement x value',float),('Displacement y value',float)])
        self.disp['Node index']=disp_nodes
        self.disp['Displacement x is prescribed']=disp_bools_x
        self.disp['Displacement y is prescribed']=disp_bools_y
        self.disp['Displacement x value']=disp_values_x
        self.disp['Displacement y value']=disp_values_y

        self._update_bc_set('displacements')
        self._update_noFail('displacements')

    def set_velocities_constraints(self,mesh:pd.Mesh,
    vel_nodes:np.ndarray|list,vel_bools_x:np.ndarray|list,vel_bools_y:np.ndarray|list,vel_values_x:np.ndarray|list,vel_values_y:np.ndarray|list) -> None:
        '''Input:

        - mesh: peridynamic mesh generated with Mesh
        - vel_nodes: which nodes have constrained velocities | type np.array or list of ints
        - vel_bools_x: wheter the velocitity in the x direction is prescribed | type: np.array or list of bools
        - vel_bools_y: wheter the velocitity in the y direction is prescribed | type: np.array or list of bools
        - vel_values_x: values of prescribed velocitity in the x direction | type: np.array or list of ints
        - vel_values_y: values of prescribed velocitity in the y direction | type: np.array or list of ints

        Output:

        Updates class attribute vel with a structured np.array:
        [Node index, Velocitity x is prescribed, Velocitity y is prescribed, Velocitity x value, Velocitity y value]
        
        Where:

        - Node index: which nodes have constrained velocities | type: np.array of ints
        - Velocitity x is prescribed: wheter the velocitity in the x direction is prescribed | type: np.array of bools
        - Velocitity y is prescribed: wheter the velocitity in the y direction is prescribed | type: np.array of bools
        - Velocitity x value: values of prescribed velocitity in the x direction | type: np.array of ints
        - Velocitity y value: values of prescribed velocitity in the y direction | type: np.array of ints'''

        self._mesh=mesh

        self.vel=np.empty_like(vel_nodes,
            dtype=[('Node index',int),('Velocity x is prescribed',bool),('Velocity y is prescribed',bool),
            ('Velocity x value',float),('Velocity y value',float)])
        self.vel['Node index']=vel_nodes
        self.vel['Velocity x is prescribed']=vel_bools_x
        self.vel['Velocity y is prescribed']=vel_bools_y
        self.vel['Velocity x value']=vel_values_x
        self.vel['Velocity y value']=vel_values_y

        self._update_bc_set('velocities')
        self._update_noFail('velocities')

    def _update_noFail(self,for_what:str) -> None:

        # Initialize the noFail np.array
        if self.noFail.size==0:
            self.noFail=np.full((len(self._mesh.points)),False)

        # Add "force constraints", when x or y forces are 0
        if for_what=='bodyforces':
            self.noFail[np.logical_or(self.b[:,0]!=0,self.b[:,1]!=0)]=True

        # Add displacement constraint nodes to the noFail variable
        elif for_what=='displacements':
            self.noFail[self.disp['Node index']]=True

        # Add velocity constraint nodes to the noFail variable
        elif for_what=='velocities':
            self.noFail[self.vel['Node index']]=True
        
    def _update_bc_set(self,for_what:str) -> None:

        # Initialize the bc_set np.array
        if self._bc_set.size==0:
            self._bc_set=np.full((2*len(self._mesh.points),3),np.nan,dtype=float)

        # Update the displacement constraints
        if for_what=='displacements':
            disp_2D_to_1D=np.hstack((
            2*self.disp['Node index'][self.disp['Displacement x is prescribed']==True],
            2*self.disp['Node index'][self.disp['Displacement y is prescribed']==True]+1
            ))
            self._bc_set[disp_2D_to_1D,0]=disp_2D_to_1D
            values_2D_to_1D=np.hstack((
            self.disp['Displacement x value'][self.disp['Displacement x is prescribed']==True],
            self.disp['Displacement y value'][self.disp['Displacement y is prescribed']==True]
            ))
            self._bc_set[disp_2D_to_1D,1]=values_2D_to_1D

        # Update the velocities constraints
        elif for_what=='velocities':
            vel_2D_to_1D=np.hstack((
            2*self.vel['Node index'][self.vel['Velocity x is prescribed']==True],
            2*self.vel['Node index'][self.vel['Velocity y is prescribed']==True]+1
            ))
            self._bc_set[vel_2D_to_1D,0]=vel_2D_to_1D
            values_2D_to_1D=np.hstack((
            self.vel['Velocity x value'][self.vel['Velocity x is prescribed']==True],
            self.vel['Velocity y value'][self.vel['Velocity y is prescribed']==True]
            ))
            self._bc_set[vel_2D_to_1D,2]=values_2D_to_1D

        # Doesn't do anything
        elif for_what=='bodyforces':
            pass

    @_cached_property
    def _constrained_meshodes_bool(self):

        if self._bc_set.size>0:
            constrained_meshodes_bool=~np.isnan(self._bc_set[:,0])

        return constrained_meshodes_bool
    
    @_cached_property
    def bc_set(self):

        return self._bc_set[self._constrained_meshodes_bool]

    @_cached_property
    def ndof(self):
        
        # Number of deggrees of freedom

        return 2*len(self._mesh.points)-len(self.bc_set)

    @_cached_property
    def idb(self):

        # DEFINE THE IDB VECTOR
        idb=np.empty((2*len(self._mesh.points)),dtype=int)
        try:
            # Constrained degree of freedom
            # Gets bc_set first collumn, which contains the constrained nodes indices
            #id_const=self.bc_set[:,0].astype(int)
            idb[self._constrained_meshodes_bool]=np.arange(self.ndof,self.ndof+np.count_nonzero(self._constrained_meshodes_bool))
            # Free degree of freedom
            # Get indices that are not constrained
            #id_dof=np.setdiff1d(np.arange(0,len(idb)),id_const,assume_unique=True)
            idb[~self._constrained_meshodes_bool]=np.arange(0,np.count_nonzero(~self._constrained_meshodes_bool)) 
        except IndexError:
            pass

        return idb

    @_cached_property
    def dofi(self):

        # i=np.arange(0,2*len(self._mesh.points))
        # dofi=np.column_stack((self.idb[i[::2]],self.idb[i[1::2]]))

        return self.idb.reshape((-1,2))

    @_cached_property
    def bb(self):

        if self.b.size==0:
            return np.array([])

        bb=np.empty((2*len(self._mesh.points)),dtype=float)
        bb[self.dofi]=self.b

        return bb

    def compute_dofj(self,family:pd.Family) -> None:
        '''Compute dofj as a class attribute'''

        self.dofj=[]
        for j in family.j:
            self.dofj.append(np.column_stack((self.idb[2*j],self.idb[2*j]+1)))

        self.dofj=np.array(self.dofj,dtype=object)

    def compute_noFail_ij(self,family:pd.Family) -> None:

        self.noFail_ij=[]
        for i in range(0,len(family.j)):
            self.noFail_ij.append(np.logical_or(self.noFail[i],self.noFail[family.j[i]]))
        self.noFail_ij=np.array(self.noFail_ij,dtype=object)

    def set_damage_on(self,true_or_false:bool) -> None:
        if type(true_or_false)!=bool:
            raise UserWarning('damage_on should be a boolean')
        self.damage.damage_on=true_or_false

    def set_damage_dependent_SC(self,true_or_false:bool) -> None:
        if type(true_or_false)!=bool:
            raise UserWarning('damage_on should be a boolean')
        self.damage.damage_dependent_Sc=true_or_false

    def set_crackIn(self,Nx2_list_or_array:np.ndarray|list) -> None:
 
        if type(Nx2_list_or_array)==list:
            Nx2_list_or_array=np.array(Nx2_list_or_array)

        try:    
            if Nx2_list_or_array.shape[1]!=2:
                raise UserWarning('Crack segments must be a (N,2) shape')
        except IndexError:
            raise UserWarning('Crack segments must have a (N,2) shape')
        if len(Nx2_list_or_array)<2:
            raise UserWarning('The crack segments must have at least 2 points')

        self.damage.crackIn=Nx2_list_or_array

    def initialize_brokenBonds(self,family:pd.Family) -> None:

        # try:
        #     crackSegments=len(self.damage.crackIn)
        # except AttributeError:
        #     raise UserWarning('Pleas set the initial damage crackIn by using the metdos set_damage_crackIn()')

        self.damage.brokenBonds=[]
        for j in family.j:
            #self.damage.brokenBonds.append(np.full((len(j),crackSegments-1),False,dtype=bool))
            self.damage.brokenBonds.append(np.full((len(j)),False,dtype=bool))

        self.damage.brokenBonds=np.array(self.damage.brokenBonds,dtype=object)

    def plot(self,fig_max_size_cm:tuple[float|int,float|int]=[40.64,40.64],markersize:int=6,
    free_nodes_marker:str='o',free_nodes_color:str='black',
    traction_nodes_marker:str='s',traction_nodes_color:str='blue',
    disp_nodes_marker:str='+',disp_nodes_color:str='red',
    vel_nodes_marker:str='x',vel_nodes_color:str='green',
    crackIn_color:str='m',crackIn_width:int=4,**kwargs) -> None:
        '''Plots the boundary conditions set. Can also plot initial damage crack if damage is supplied by using the set_crackIn method'''

        # Initiate the matplotlib figure and axis
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=np.array(fig_max_size_cm)/2.54)
        
        # Parameters to reduce margins and fix aspect ratio
        ax.set_aspect(1)
        # ax.set_frame_on(frame_on) # removes the black rectangle if fram_on=False
        # ax.grid(grid)
        
        # Plot the data
        ax.plot(self._mesh.x,self._mesh.y,linestyle='',marker=free_nodes_marker,color=free_nodes_color,label='Free nodes',markersize=markersize,zorder=1)
        if self.b.size>0:
            if self.b.ndim==2:
                ax.plot(self._mesh.x[self.bodyForce['Node index']],\
                        self._mesh.y[self.bodyForce['Node index']],\
                        linestyle='',marker=traction_nodes_marker,color=traction_nodes_color,markersize=markersize,label='Traction force nodes',zorder=2)
            else:
                ax.plot(self._mesh.x[np.logical_or(self.b[-1,:,0]!=0,self.b[-1,:,1]!=0)],\
                        self._mesh.y[np.logical_or(self.b[-1,:,0]!=0,self.b[-1,:,1]!=0)],\
                        linestyle='',marker=traction_nodes_marker,color=traction_nodes_color,markersize=markersize,label='Traction force nodes',zorder=2)
        if self.bc_set.size>0:
            if self.disp.size>0:
                ax.plot(self._mesh.x[self.disp['Node index']],\
                        self._mesh.y[self.disp['Node index']],\
                        linestyle='',marker=disp_nodes_marker,color=disp_nodes_color,markersize=markersize,label='Disp. constraint nodes',zorder=2)
            if self.vel.size>0:
                ax.plot(self._mesh.x[self.vel['Node index']],\
                        self._mesh.y[self.vel['Node index']],\
                        linestyle='',marker=vel_nodes_marker,color=vel_nodes_color,markersize=markersize,label='Velocity nodes',zorder=2)
        if self.damage.crackIn is not None:
            ax.plot(self.damage.crackIn[:,0],self.damage.crackIn[:,1],\
                    linestyle='-',marker='',color=crackIn_color,markersize=markersize,label='Initial crack',linewidth=crackIn_width,zorder=2)
        
        # Temporarily removes the margins to get the correct figure size in pixels
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        # Updates the figure so it knows each artist's size
        fig.draw_without_rendering()

        # Calculate the largest marker size in user units instead of pixels
        # [len x (user unit), len y (user unit)]/[len x (pixels), len y (pixels)]
        # unit_per_px=[max(self._mesh.x)-min(self._mesh.x),max(self._mesh.y)-min(self._mesh.y)]/ax.bbox.size
        # largest_marker_size=6*unit_per_px
        largest_marker_size=np.max([markersize,crackIn_width])*np.min(pd._plot_helpers.get_figure_relative_unit('unit/pt',ax,self._mesh.x.max()-self._mesh.x.min(),self._mesh.y.max()-self._mesh.y.min()))

        ax.set_title('Boundary conditions',{'fontsize':24}) # e.g. Mesh
        ax.set_xlabel(f'x ({self._mesh.unit})',fontsize=18) # e.g. x (m)
        ax.set_ylabel(f'y ({self._mesh.unit})',fontsize=18) # e.g. y (m)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1, 0, 0),framealpha=1,edgecolor='grey',fontsize=18)#,facecolor='white'
        
        # Apply corrected limits
        ax.set_xlim((min(self._mesh.x)-largest_marker_size,max(self._mesh.x)+largest_marker_size))
        ax.set_ylim((min(self._mesh.y)-largest_marker_size,max(self._mesh.y)+largest_marker_size))

        # Make sure the final layout is correctly rendered
        fig.draw_without_rendering()

        # Adjust the final figure size according to the axis size
        if ax.get_tightbbox().size[0]>ax.get_tightbbox().size[1]:
            # x is bigger
            new_size=ax.get_tightbbox().size*fig.get_size_inches()[0]/ax.get_tightbbox().size[0]
        else:
            # y is bigger
            new_size=ax.get_tightbbox().size*fig.get_size_inches()[1]/ax.get_tightbbox().size[1]

        fig.set_size_inches(new_size)
        # fig.set_constrained_layout(True)
        # fig.set_constrained_layout_pads(w_pad=2/fig.get_dpi(),h_pad=0,wspace=0,hspace=0)
        fig.draw_without_rendering()

        try:
            if kwargs['hide_plot_and_update_class_properties']==True:
                self._fig=fig
                self._ax=ax
                plt.close()
                return
        except KeyError:
            plt.show()
        
    def get_fig_ax(self,**kwargs) -> None:

        '''Returns matplotlib figure object. See the plot method of this class for optional parameters'''

        self.plot(hide_plot_and_update_class_properties=True,**kwargs)
        
        return self._fig,self._ax
