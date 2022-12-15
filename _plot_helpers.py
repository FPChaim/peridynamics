'''Provides some functions related to plotting'''

import os
from time import strftime
import matplotlib.pyplot as plt
from matplotlib import rcsetup
from PIL import Image

def get_figure_relative_unit(which,ax,x_interval_in_unit=None,y_interval_in_unit=None):

    '''Input parameters:
    
    - which:
        - "px/unit": pixels per unit | Usage --> unit_x * px_per_unit[0] = px
        - "unit/px": units per pixel | Usage --> px * unit_per_px[0] = units
        - "in/unit": inches per unit | Usage --> unit_x * in_per_unit[0] = in
        - "unit/in": units per inch | Usage --> in * unit_per_ni[0] = units
        - "pt/unit": points per unit. Note: 1 pt = 1/72 in | Usage --> unit_x * pt_per_unit[0] = pt
        - "unit/pt": units per point. Note: 1 pt = 1/72 in  | Usage --> pt * unit_per_pt[0] = units
        - "pt²/unit": points² per unit. Notes: 1 pt² = 1/5184 in. Usefull for scatter plot sizing | Usage --> unit_x * pt2_per_unit[0] = pt2
        - "unit/pt²": units per points². Notes: 1 pt² = 1/5184 in. Usefull for scatter plot sizing  | Usage --> pt² * unit_per_pt2[0] = units
    - ax: matplotlib axis related to a 2D figure (note: changing figure size invalides the relation)
    - x_interval_in_unit: (xmax+margin)-(xmin+margin) or xmax-xmin if ax.ax.set_xmargin(0) was used or None to automatically get already plotted x data on axis
    - y_interval_in_unit: (ymax+margin)-(ymin+margin) or ymax-ymin if ax.ax.set_ymargin(0) was used or None to automatically get already plotted y data on axis
    
    Output parameters:
    
    - (2,) array of choosen relation'''

    # from numpy import diff,all,isinf

    fig=ax.get_figure()

    if x_interval_in_unit==None:
        #x_interval_in_unit=ax.xaxis.get_data_interval()[1]-ax.xaxis.get_data_interval()[0] # get_data_interval is the same as xmax-xmin. Must use ax.ax.set_xmargin(0) on figure before using
        x_interval_in_unit=ax.get_xlim()[1]-ax.get_xlim()[0] # xlim is the same as xmax+margin-(xmin+margin)

    if y_interval_in_unit==None:
        #y_interval_in_unit=ax.yaxis.get_data_interval()[1]-ax.yaxis.get_data_interval()[0]
        y_interval_in_unit=ax.get_ylim()[1]-ax.get_ylim()[0] # ylim is the same as (ymax+margin)-(ymin+margin)

    if which=='px/unit':
        return ax.bbox.size/[x_interval_in_unit,y_interval_in_unit]

    elif which=='unit/px':
        return [x_interval_in_unit,y_interval_in_unit]/ax.bbox.size

    elif which=='in/unit':
        return (ax.bbox.size/fig.dpi)/[x_interval_in_unit,y_interval_in_unit]

    elif which=='unit/in':
        return [x_interval_in_unit,y_interval_in_unit]/(ax.bbox.size/fig.dpi)
    
    elif which=='pt/unit':
        # return (ax.bbox.size/fig.dpi)/[x_interval_in_unit,y_interval_in_unit]*72
        return (ax.bbox.size/(fig.dpi/72))/[x_interval_in_unit,y_interval_in_unit] # note: 1 pt = fig.dpi/72 ppi --> 1 pt = fig.dpi/72 pixels | 1 pt = 1/72 in --> 1 in = 72 pt

    elif which=='unit/pt':
        # return [x_interval_in_unit,y_interval_in_unit]/(ax.bbox.size/fig.dpi)/72
        return [x_interval_in_unit,y_interval_in_unit]/(ax.bbox.size/(fig.dpi/72)) # note: 1 pt = fig.dpi/72 ppi --> 1 pt = fig.dpi/72 pixels | 1 pt = 1/72 in --> 1 in = 72 pt

    # elif which=='pt²/unit':
    #     return (ax.bbox.size/fig.dpi)/[x_interval_in_unit,y_interval_in_unit]*5184 # note: 1 point² = 1/5184 in --> 1 in = 5184 pt²

    # elif which=='unit/pt²':
    #     return [x_interval_in_unit,y_interval_in_unit]/(ax.bbox.size/fig.dpi)/5184 # note: 1 point² = 1/5184 in --> 1 in = 5184 pt²

    else:
        raise NameError('Available options are: "px/unit", "unit/px", "in/unit", "unit/in", "pt/unit", "unit/pt"')

def is_notebook() -> bool:
    '''Checks if the code is running from within a notebook.
    From https://stackoverflow.com/a/39662359'''

    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        pass

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False # Terminal running IPython
        else:
            return False # Other type (?)
    except NameError:
        return False # Probably standard Python interpreter

# is_notebook()

# User configuration: static or interactive plots

# Static plots
# % matplotlib inline
# plt.switch_backend('module://matplotlib_inline.backend_inline')

# Interactive plots
# %matplotlib ipympl
# %matplotlib widget
# plt.switch_backend('module://ipympl.backend_nbagg')

def matplotlib_backend_is_GUI() -> bool:
    current_backend=plt.get_backend()
    interactive_backends=rcsetup.interactive_bk # gui backends, can be shown by plt.show()
    non_interactive_backends=rcsetup.non_interactive_bk # non-GUI backend. Can't be shown by plt.show()
    notebook_backends=['module://matplotlib_inline.backend_inline', # gui backends, can be shown by plt.show()
                       'module://ipympl.backend_nbagg']

    if interactive_backends.count(current_backend)==1 or notebook_backends.count(current_backend)==1:
        return True
    elif non_interactive_backends.count(current_backend)==1:
        return False
    else:
        raise NotImplementedError(f'The backend "{current_backend}" is not recognized')

# matplotlib_backend_is_GUI()

def save_plot(fig,ax,name='%Y-%m-%d-%t.png',folder='',transparent=False,silent=False,dpi=100,pad_px=3):
    r'''- fig: matplotlib figure
    - ax: matplotlib axis
    - name: Can be any string constructed with the following codes:
        Plot codes:
            %t  Plot title
        Time codes (from time.strftime library):
            %Y  Year with century as a decimal number.
            %m  Month as a decimal number [01,12].
            %d  Day of the month as a decimal number [01,31].
            %H  Hour (24-hour clock) as a decimal number [00,23].
            %M  Minute as a decimal number [00,59].
            %S  Second as a decimal number [00,61].
            %z  Time zone offset from UTC.
            %a  Locale's abbreviated weekday name.
            %A  Locale's full weekday name.
            %b  Locale's abbreviated month name.
            %B  Locale's full month name.
            %c  Locale's appropriate date and time representation.
            %I  Hour (12-hour clock) as a decimal number [01,12].
            %p  Locale's equivalent of either AM or PM.
        Other codes may be available on your platform.  See documentation for
        the C library strftime function.
        Example: name = '%Y-%m-%d-%t-my custom text' will wield a name with
        the following characteristics = YYYY-MM-DD-FIGURETITLE-my custom text
    - extension: figure extension
    - folder: set the folder to where the images will be saved. By default
    creates and the folder 'Peridynamic plots' where the script is running and
    set it for saving the images.
    Example: folder = r'C:\Users\USERNAME\Desktop\Peridynamic plots'
    - transparent: wheter the image will be trasnparent or not (only supported on a few extensions)
    - silent: wheter to print a message containg the full image path of created image
    - dpi: density per inches. Higher dpi means a better image with bigger file size
    - pad_px: padding around the figure (0 is not recommended)'''
    
    if folder=='':
        folder=os.path.join(os.getcwd(),'Peridynamic plots')
    
    if os.path.isdir(folder)==False: # creates folder if it doesn't exist
        os.mkdir(folder)
     
    # for i,j in zip(['%t'],[ax.get_title()]): # custom name formatting with plot attributes
    #     name=name.replace(i,j)
    name=name.replace('%t',ax.get_title()) # custom name formatting with plot attributes
    
    name=strftime(name) # name formatting with time codes

    absolute_path=os.path.join(folder,name)
    
    # if transparent==True:
    #     bbox_inches=None
    #     facecolor=(1,1,1,0) # (R,G,B,A)
    #     edgecolor=(1,1,1,0) # (R,G,B,A)

    # else:
    #     bbox_inches='tight'
    #     facecolor=(1,1,1,1) # (R,G,B,A)
    #     edgecolor=(1,1,1,1) # (R,G,B,A)
    
    # fig.savefig(fname=absolute_path,transparent=transparent,dpi=dpi,
    #            bbox_inches=bbox_inches,pad_inches=pad_px/dpi,facecolor=facecolor,edgecolor=edgecolor)

    fig.savefig(fname=absolute_path,transparent=transparent,dpi=dpi,
               bbox_inches='tight',pad_inches=pad_px/dpi)
    
    # if transparent==True:
    #     im=Image.open(absolute_path)
    #     im=im.crop(im.getbbox()) # crops extra transparent parts
    #     im.save(absolute_path)
    
    if silent==False:
        print(f'Plot succesfully saved at {absolute_path}')
        
# save_plot(fig,ax,folder=r'C:\Users\FPChaim\Desktop\Peridynamic plots',transparent=True)