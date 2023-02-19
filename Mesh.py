'''Provides all the necessary Mesh code necessary for the peridynamics module'''

from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
from peridynamics._plot_helpers import get_figure_relative_unit

class Mesh:
    '''Class that should initiate instances to create or update a peridynamic mesh utilizing different built-in methods
    Usage example:
    mesh = Mesh() # Initiates the class instance
    mesh.generate_rectangular_mesh(h=2,size_x=6,size_y=4) # updates the mesh variable attributes
    mesh.center()
    print(f'Mesh points are =\\n{mesh.points}')
    print(f'The grid spacing is = {mesh.h}')
    print(f"The elements' area are = {mesh.A}")
    mesh.plot()'''
    
    def __init__(self):
        self.rotation_angle_rad=0
        self.type='none'
        self.unit='m'

    def set_unit(self,unit):
        '''Only changes what's displayed on the labels. The user is responsible for any unit conversions.'''
        self.unit=unit
        
    @cached_property
    def x(self):
        '''Internal method to get x coordinates from the class' points attribute'''
        try:
            return self.points[:,0]
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
    
    @cached_property
    def y(self):
        '''Get y coordinates from the class' points attribute'''
        try:
            return self.points[:,1]
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')

    @cached_property
    def grid(self):
        '''Returns a list with [X,Y], np.where:
            - X: nrows x ncols dimension np.array np.where each row represent each x value of given row
            - Y: nrows x ncols dimension np.array np.where each row represent each y value of given row
            Only works for rectangular meshs'''

        ϵ=self.h/2
        X,Y=np.meshgrid(np.arange(self.x.min(),self.x.max()+ϵ,self.h),np.arange(self.y.min(),self.y.max()+ϵ,self.h))
        return X,Y
            
    @cached_property
    def ncols(self):
        '''Get the number of columns'''
        
        try:
            lenghtx=max(self.x)-min(self.x)
            if self.rotation_angle_rad==0:
                cols=int(lenghtx/self.h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
                [rem_1,add_1]=np.isclose(lenghtx/self.h+1,[cols-1,cols+1],rtol=1.e-5,atol=1.e-8)
                if rem_1==True:
                    cols=cols-1
                elif add_1==True:
                    cols=cols+1
                return cols
            else:
                cols=int(lenghtx/self.h*np.cos(self.rotation_angle_rad))+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
                [rem_1,add_1]=np.isclose(lenghtx/self.h*np.cos(self.rotation_angle_rad)+1,[cols-1,cols+1],rtol=1.e-5,atol=1.e-8)
                if rem_1==True:
                    cols=cols-1
                elif add_1==True:
                    cols=cols+1
                return cols
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
            
    @cached_property
    def nrows(self):
        '''Get the number of rows'''

        try:
            lenghty=max(self.y)-min(self.y)
            if self.rotation_angle_rad==0:
                rows=int(lenghty/self.h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
                [rem_1,add_1]=np.isclose(lenghty/self.h+1,[rows-1,rows+1],rtol=1.e-5,atol=1.e-8)
                if rem_1==True:
                    rows=rows-1
                elif add_1==True:
                    rows=rows+1
                return rows
            else:
                rows=int(lenghty/self.h*np.sin(self.rotation_angle_rad))+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
                [rem_1,add_1]=np.isclose(lenghty/self.h*np.sin(self.rotation_angle_rad)+1,[rows-1,rows+1],rtol=1.e-5,atol=1.e-8)
                if rem_1==True:
                    rows=rows-1
                elif add_1==True:
                    rows=rows+1
                return rows
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')

    @cached_property
    def counters(self):

        # 3 counters: tip
        # 4 counters: border
        # 5 counters: border
        # 6 counters: border
        # 7 counters: border
        # 8 counters: fully inside

        radius=self.h*1.5 # diagonal is self.h*sqrt(2), so whe choose a value that is a little bit higher to avoid float errors
        counters=np.empty((len(self.points)),dtype=int)
        for i in range(0,len(self.points)):
            dist_xi=np.linalg.norm(self.points[i]-self.points,axis=1)
            counters[i]=np.count_nonzero(dist_xi<radius)-1 # -1 to exclude self
        
        return counters

    @cached_property
    def _border_bool(self):

        all=self.counters<8
        top=self.y>self.y.max()-self.h/2
        left=self.x<self.x.min()+self.h/2
        right=self.x>self.x.max()-self.h/2
        bottom=self.y<self.y.min()+self.h/2

        dictionary={'all':all,
            'top':top,
            'left':left,
            'right':right,
            'bottom':bottom}

        return dictionary

    def get_borders_bool(self,which='all'):
        '''Get borders booleans. Also see: get_borders_ind and get_borders'''

        if which=='all':
            bool_array=self._border_bool[which]
        else:
            if self.type=='rectangular':
                bool_array=self._border_bool[which]
            else:
                raise NotImplementedError('"top", "left", "right" and "bottom" are not supported for non rectangular meshs')

        return bool_array

    def get_borders_ind(self,which='all'):
        '''Get borders indices. Also see: get_borders_bool and get_borders_ind'''

        if which=='all':
            ind_array=np.where(self._border_bool[which])[0]
        else:
            if self.type=='rectangular':
                ind_array=np.where(self._border_bool[which])[0]
            else:
                raise NotImplementedError('"top", "left", "right" and "bottom" are not supported for non rectangular meshs')

        return ind_array

    def get_borders(self,which='all'):
        '''Get borders coordinates. Also see: get_borders_bool and get_borders_ind'''

        if which=='all':
            border_array=self.points[self._border_bool[which]]
        else:
            if self.type=='rectangular':
                border_array=self.points[self._border_bool[which]]
            else:
                raise NotImplementedError('"top", "left", "right" and "bottom" are not supported for non rectangular meshs')

        return border_array

    def generate_rectangular_mesh(self,h,size_x,size_y,mode='restrictive'):
        '''Creates or updates the following class attributes:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: numpy.ndnp.array
        - h: grid spacing | type: float or int
        - A: elements' area | type: float or int
        - points: the points' [x,y] positions
        - vertices: rectangle vertices positions
        - x: position x of the points
        - y: position y of the points
        Input:
        - h: grid spacing | type: float or int
        - size_x: size of the rectangular mesh side on the x axis | type: float or int
        - size_y: size of the rectangular mesh side on the y axis | type: float or int
        - mode: determine how the points fit inside the given rectangular shape | type: str
            - "restrictive": make sure the maximum values of points don't surpass the values imposed by the sides, allowing them to be equal or smaller by h
            - "expansive": allows the maximum values of points to be equal or surpass the values of the sides by a maximum of h
            - "exact": make sure the points fit perfectly into the rectangular shape. Tecnically this can be done by the other modes, but this one raises a warning in case the user inputs wrong parameters'''
        
        if h>size_x or h>size_y:
            raise ValueError("Grid spacing can't be higher than any of the rectangle sizes")
        
        if h==0:
            raise ValueError("Grid spacing can't be 0")
        
        if size_x<=0 or size_y<=0:
            raise ValueError("The sides of the rectangle must be a positive, non-zero value")
        
        self.h=h
        self.A=h**2 # elements' area
        self.type='rectangular'

        cols=int(size_x/h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
        [rem_1,add_1]=np.isclose(size_x/h+1,[cols-1,cols+1],rtol=1.e-5,atol=1.e-8)
        if rem_1==True:
            cols=cols-1
        elif add_1==True:
            cols=cols+1

        rows=int(size_y/h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
        [rem_1,add_1]=np.isclose(size_y/h+1,[rows-1,rows+1],rtol=1.e-5,atol=1.e-8)
        if rem_1==True:
            rows=rows-1
        elif add_1==True:
            rows=rows+1
        
        if mode=='restrictive':
            pass
        elif mode=='expansive':
            size_x=size_x+h
            size_y=size_y+h
            cols=cols+1
            rows=rows+1
        elif mode=='exact':
            if ~(np.isclose(cols,size_x/h) and ~np.isclose(rows,size_y/h)):
                raise ValueError('In "exact" mode the h must be a divisor of both sides')
        else:
            raise NameError('Available modes are "restrictive", "expansive" and "exact"')

        x=np.arange(0,cols)*h
        y=np.arange(0,rows)*h

        x=np.resize(x,(rows*cols))

        y=np.resize(y,(rows*cols))
        y.sort() # sort y so yi will match every xi. Mesh points order will be from lower left growing to the left, then up

        points=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        
        self.vertices=np.array([[0,0],[size_x,0],[size_x,size_y],[0,size_y],[0,0]])
        self.points=points

    def generate_mesh_constrained_within_points(self,h,points_Nx2):
        '''Generate mesh inside the given points. Points should be in order.
        Creates or updates the following class attributes:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: numpy.ndnp.array
        - h: grid spacing | type: float or int
        - A: elements' area | type: float or int
        Input:
        - h: grid spacing | type: float or int
        - points_Nx2: ordered polygon vertices in Nx2 dimension | type: float or int'''
        
        self.h=h
        self.A=h**2 # elements' area
        
        if type(points_Nx2)==list:
            points_Nx2=np.array(points_Nx2)
        
        if np.all(points_Nx2[-1]!=points_Nx2[0]): # make sure points form a closed polygon
            points_Nx2=np.vstack((points_Nx2,points_Nx2[0]))
        
        [xi,yi]=np.min(points_Nx2,axis=0)
        [xf,yf]=np.max(points_Nx2,axis=0)
        
        size_x=xf-xi
        size_y=yf-yi

        cols=int(size_x/h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
        [rem_1,add_1]=np.isclose(size_x/h+1,[cols-1,cols+1],rtol=1.e-5,atol=1.e-8)
        if rem_1==True:
            cols=cols-1
        elif add_1==True:
            cols=cols+1

        rows=int(size_y/h)+1 # can cause errors when the division is close to an integer, eg 3.999999 or 4.000001
        [rem_1,add_1]=np.isclose(size_y/h+1,[rows-1,rows+1],rtol=1.e-5,atol=1.e-8)
        if rem_1==True:
            rows=rows-1
        elif add_1==True:
            rows=rows+1

        x=np.arange(0,cols)*h
        y=np.arange(0,rows)*h

        x=np.resize(x,(rows*cols))

        y=np.resize(y,(rows*cols))
        y.sort() # sort y so yi will match every xi. Mesh points order will be from lower left growing to the left, then up

        points=np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        
        path=Path(vertices=points_Nx2,closed=True)
        points_inside=path.contains_points(points) # checks what points are inside
        
        self.vertices=np.array(points_Nx2)
        self.points=points[points_inside]
        
    def center(self):
        '''Offsets mesh so it's centered on the plane origin.
        Updates the following class attribute:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: np.array'''
        
        try:
            self.points
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
        
        self.vertices=self.vertices-[(min(self.points[:,0])+max(self.points[:,0]))/2,(min(self.points[:,1])+max(self.points[:,1]))/2]
        self.points=self.points-[(min(self.points[:,0])+max(self.points[:,0]))/2,(min(self.points[:,1])+max(self.points[:,1]))/2]
        
    def offset(self,x,y):
        '''Offsets mesh to by x,y ammount.
        Updates the following class attribute:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: np.array
        Input:
        - x: distance in the x axis | type: float or int
        - y: distance in the y axis | type: float or int'''
        
        try:
            self.points
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
        
        self.vertices=self.vertices+[x,y]
        self.points=self.points+[x,y]
        
    def rotate(self,angle,unit='deg'):
        '''Rotates mesh around origin.
        Updates the following class attribute:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: np.array
        Input:
        - angle: counterclockwise mesh rotation angle in degrees (default) or radians | type: float or int
        - unit: wheter the angle should be in degrees ("deg") or radians ("rad") | type: str'''

        try:
            self.points
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
                
        if unit=='deg':
            angle=np.deg2rad(angle)
        elif unit=='rad':
            pass
        else:
            raise NameError('Available unit options are "deg" and "rad"')
        
        aux1=np.copy(self.vertices)
        self.vertices[:,0]=aux1[:,0]*np.cos(angle)-aux1[:,1]*np.sin(angle)
        self.vertices[:,1]=aux1[:,1]*np.cos(angle)+aux1[:,0]*np.sin(angle)
        
        aux2=np.copy(self.points)
        self.points[:,0]=aux2[:,0]*np.cos(angle)-aux2[:,1]*np.sin(angle)
        self.points[:,1]=aux2[:,1]*np.cos(angle)+aux2[:,0]*np.sin(angle)
        
        self.rotation_angle_rad=angle
        
    def remove_circles(self,centers,radiuses):
        '''Remove all points inside circles
        Updates the following class attribute:
        - points: [[x1,y1],[x2,y2],...,[xn,yn]] points that represents the peridynamic mesh | type: np.array
        Input:
        - centers: center of the circles | type: list
        - radiuses: radiuses of each circle | type: list'''
        
        try:
            self.points
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
             
        # Remove points but keep vertices    
        for center,radius in zip(centers,radiuses):
            r=np.linalg.norm(self.points-center,ord=2,axis=1) # euclidian distance between each pair of points
            self.points=self.points[np.where(r>radius)[0]]
            
    def plot(self,include_nodes=True,include_cells=True,include_vertices=False,include_sides=False,
                frame_on=False,grid=False,fig_max_size_cm=[40.64,40.64],
                legend_loc='lower right',legend_bbox_to_anchor=(1,1,0,0),legend_framealpha=1,legend_edgecolor='grey',
                title='Mesh',nodes_size=0.5,nodes_color='b',cells_color='grey',
                vertices_linestyle='',vertices_marker='o',vertices_markersize=12,vertices_color='black',
                sides_linestyle='-',sides_marker='',sides_linewidth=6,sides_color='black',**kwargs):
        '''Plot the mesh'''
        
        try:
            self.points
        except AttributeError:
            raise UserWarning('A mesh should be generated first. Type help(Mesh) to get started.')
        
        # Initiate the matplotlib figure and axis
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=np.array(fig_max_size_cm)/2.54)#,subplot_kw={'projection':None},gridspec_kw={'wspace':0,'hspace':0}

        # Parameters to reduce margins and fix aspect ratio
        ax.set_aspect(1)#,adjustable='box'
        # Temporarily removes the margins to get the correct figure relative units
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.set_frame_on(frame_on) # removes the black rectangle if frame_on=False
        ax.grid(grid)

        # Plot the points
        nodes_markersize_in_units=self.h*nodes_size
        cells_markersize_in_units=self.h
        if include_nodes==True:
            # ax.scatter(self.x,self.y,marker=nodes_marker,c=nodes_color,edgecolors='none',label='Nodes',
            # s=(nodes_markersize_in_units*min(get_figure_relative_unit('pt/unit',ax,self.x.max()-self.x.min(),self.y.max()-self.y.min())))**2,
            # linewidths=1)
            for i in self.points:
                ax.add_patch(patches.Circle(i,nodes_markersize_in_units/2,facecolor=nodes_color,edgecolor='none',linewidth=1,zorder=2))

        if include_cells==True:
            # ax.scatter(self.x,self.y,marker='s',c='none',edgecolors=cells_color,label='Cells',
            # s=(cells_markersize_in_units*min(get_figure_relative_unit('pt/unit',ax,self.x.max()-self.x.min(),self.y.max()-self.y.min())))**2,
            # linewidths=2)
            for i in self.points:
                ax.add_patch(patches.Rectangle(i-self.h/2,self.h,self.h,facecolor='none',edgecolor=cells_color,linewidth=1))

        if include_vertices==True:
            vert,=ax.plot(self.vertices[:,0],self.vertices[:,1],linestyle=vertices_linestyle,marker=vertices_marker,
                    color=vertices_color,markersize=vertices_markersize,label='Vertices')
        if include_sides==True:
            sides,=ax.plot(self.vertices[:,0],self.vertices[:,1],linestyle=sides_linestyle,marker=sides_marker,
                    color=sides_color,linewidth=sides_linewidth,label='Sides')
    
        # Get minimun and maximum values for x and y points
        min_x=self.x.min()
        max_x=self.x.max()
        min_y=self.y.min()
        max_y=self.y.max()
        if include_vertices or include_sides==True:
            if np.min(self.vertices[:,0])<min_x:
                min_x=np.min(self.vertices[:,0])
            if np.max(self.vertices[:,0])>max_x:
                max_x=np.max(self.vertices[:,0])
            if np.min(self.vertices[:,1])<min_y:
                min_y=np.min(self.vertices[:,1])
            if np.max(self.vertices[:,1])>max_y:
                max_y=np.max(self.vertices[:,1])

        # Gets largest marker size
        all_markers_sizes_in_units=np.hstack((nodes_markersize_in_units,cells_markersize_in_units,\
            np.array([vertices_markersize,sides_linewidth])[[include_vertices,include_sides]]*np.min(get_figure_relative_unit('unit/pt',ax,self.x.max()-self.x.min(),self.y.max()-self.y.min()))
            ))
        largest_marker_size_in_units=np.max(all_markers_sizes_in_units)

        ax.set_title(title,{'fontsize':24}) # e.g. Mesh
        ax.set_xlabel(f'x ({self.unit})',fontsize=18) # e.g. x (m)
        ax.set_ylabel(f'y ({self.unit})',fontsize=18) # e.g. y (m)

        handles=[]
        if include_nodes==True:
            handles.append(ax.scatter([1e15],[1e15],s=(nodes_markersize_in_units*min(get_figure_relative_unit('pt/unit',ax,self.x.max()-self.x.min(),self.y.max()-self.y.min())))**2,marker='o',facecolor=nodes_color,edgecolor='none',linewidths=1,label='Nodes'))
        if include_cells==True:
            handles.append(ax.scatter([1e15],[1e15],s=(cells_markersize_in_units*min(get_figure_relative_unit('pt/unit',ax,self.x.max()-self.x.min(),self.y.max()-self.y.min())))**2,marker='s',facecolor='none',edgecolor=cells_color,linewidths=1,label='Cells'))
        if include_vertices==True:
            handles.append(vert)
        if include_sides==True:
            handles.append(sides)
        # ax.legend(loc=legend_loc,bbox_to_anchor=legend_bbox_to_anchor,framealpha=legend_framealpha,
        #                   edgecolor=legend_edgecolor,fontsize=18)
        ax.legend(handles=handles,loc=legend_loc,bbox_to_anchor=legend_bbox_to_anchor,framealpha=legend_framealpha,
                          edgecolor=legend_edgecolor,fontsize=18)
        
        # Apply corrected limits
        ax.set_xlim((min_x-largest_marker_size_in_units,max_x+largest_marker_size_in_units))
        ax.set_ylim((min_y-largest_marker_size_in_units,max_y+largest_marker_size_in_units))

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

        # Obtain the figure object (see get_figure_ax) without the need to create a redundant method
        try:
            if kwargs['hide_plot_and_update_class_properties']==True:
                self._fig=fig
                self._ax=ax
                plt.close()
                return
        except KeyError:
            plt.show()
        
    def get_fig_ax(self,**kwargs):
        '''Returns matplotlib figure object. See the plot method of this class for optional parameters'''
        self.plot(hide_plot_and_update_class_properties=True,**kwargs)
        return self._fig,self._ax
