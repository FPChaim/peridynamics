'''Provides some miscellaneous functions not directly related to peridynamics'''

import os
from time import strftime
import pickle
import numpy as np
import peridynamics as pd

def progress_printer(values:float|int,fmt:str='|%b| %p %',bar_sym:str='-',bar_sep:int=116,prec:tuple[int,int,int]=[2,2,2],left_fill:tuple[str,str,str]=['','',''],
                     right_fill_for_value:tuple[int,int]=[0,0],line_end:str='\r',hundred_percent_print:str='',check_val:bool=False,flush:bool=True) -> None:
    '''Input parameters:
    
    - values:
        - values[0]: percentual values in the interval: 0<=values<=1
        - values[1]: any kind of value
    - fmt: print string custom format. Valid codes are:
        - %b progress bar
        - %p percentual value related to the progress bar (values[0])
        - %v1 arbitrary value (values[1])
        - %v2 arbitrary value (values[2])
        
        Example: "|%b| %p %"
        
    - bar_sym: bar symbol
    - bar_sep: number of bar separations
    - prec: number of decimal numbers
    - left_fil: fill for the first two cases of a percentual values smaller than 100
    - line_end: what comes after the print. The default "\\r" is used to print "above" previous prints
    - right_fill_for_value: number of reserved space for the  value contained in values[1], counting from the right
        For printing on different lines use "\\n"
    - hundred_percent_print: special print when values=1. Usefull for ending "\r" for example
    - check_val: wheter to check if number input is correct or not
    - flush: flush the printing area, allowing for more prints to be show in a short period of time. Can cause a heavy impact on performance
    
    Output parameters:
    
    - printed text'''
    
    if np.shape(values)==():
        values=[values]
    
    if check_val==True:
        if values[0]<0 or values[0]>1:
            raise UserWarning('Input value should be in the interval: 0<=value<=1')

    bar=bar_sym*int(np.ceil(values[0]*bar_sep))
    bar=f'{bar:<{bar_sep}}'

    percent=f'{values[0]*100:.{prec[0]}f}'
    percent=f'{percent:{left_fill[0]}>{3+1+prec[0]}}'
    # 3 refers to 3 integral decimals (100), 1 to the comma and prec[0]=2 refers to 2 decimal cases
    # EX: 100.00 --> 3+1+2 = 5 digits
    
    if len(values)>0: # print only the percentual value
        for i,j in zip(['%b','%p'],[bar,percent]):
            fmt=fmt.replace(i,j)
        
    if len(values)>1:
        value=f'{values[1]:.{prec[1]}e}'
        value=f'{value:{left_fill[1]}>{right_fill_for_value[0]+1+prec[1]+4}}'

        for i,j in zip(['%v1'],[value]):
            fmt=fmt.replace(i,j)

    if len(values)>2:
        value2=f'{values[2]:.{prec[2]}e}'
        value2=f'{value2:{left_fill[2]}>{right_fill_for_value[1]+1+prec[2]+4}}'

        for i,j in zip(['%v2'],[value2]):
            fmt=fmt.replace(i,j)
    
    if values[0]==1:
        print(f'{fmt}{hundred_percent_print}')
    else:
        print(fmt,end=line_end,flush=flush)

class indices_by_conditions:
    '''Verbose class to help in getting arrays inidices that satisfies given conditions. Optional'''

    def __init__(self) -> None:
        self._indices=[]
        
    def add_condition(self,condition:tuple[bool,...]) -> None:
        '''Add one initial condition or another completeley different condition'''

        if type(condition)==list or type(condition)==bool:
            condition=np.array(condition)
        self._indices.append(condition)

    def and_condition(self,condition:tuple[bool,...]) -> None:
        '''A simple "and" condition related to the latest added condition'''

        self._indices[-1]=np.logical_and(self._indices[-1],condition)

    def or_condition(self,condition:tuple[bool,...]) -> None:
        '''A simple "or" condition related to the latest added condition'''

        self._indices[-1]=np.logical_or(self._indices[-1],condition)

    def _bool_to_ind(self) -> None:
        '''Converts booleans to indices'''

        for i in range(0,len(self._indices)):
            self._indices[i]=np.where(self._indices[i])[0]

    def get_indices(self) -> np.ndarray:
        '''Use to retrieve the array containing the wanted indices'''
        
        self._bool_to_ind()

        indices_aux=self._indices
        self._indices=np.array([],dtype=int)

        for i in indices_aux:
            self._indices=np.hstack((self._indices,i))

        self._indices=np.unique(self._indices)

        return self._indices

def save_object(obj:pd.Mesh|pd.Family|pd.BoundaryConditions|pd.general_functions.Energy|np.ndarray|float|int|str|list|tuple,name:str='%Y-%m-%d-%v.pickle',folder:str='',silent:bool=False) -> None:
    r'''Input parameters:

    - obj: variable. Can be an object
    - name: Can be any string constructed with the following codes:
        - Plot codes:
            - %v  Object type (only supported for Mesh, Family, BoundaryConditions, Model and Energy peridynamic variables)
            - Will be ignored if used on an unsuported variable
        - Time codes (from time.strftime library):
            - %Y  Year with century as a decimal number.
            - %m  Month as a decimal number [01,12].
            - %d  Day of the month as a decimal number [01,31].
            - %H  Hour (24-hour clock) as a decimal number [00,23].
            - %M  Minute as a decimal number [00,59].
            - %S  Second as a decimal number [00,61].
            - %z  Time zone offset from UTC.
            - %a  Locale's abbreviated weekday name.
            - %A  Locale's full weekday name.
            - %b  Locale's abbreviated month name.
            - %B  Locale's full month name.
            - %c  Locale's appropriate date and time representation.
            - %I  Hour (12-hour clock) as a decimal number [01,12].
            - %p  Locale's equivalent of either AM or PM.

        Other codes may be available on your platform.  See documentation for
        the C library strftime function.

        Example: name = '%Y-%m-%d-%t-my custom text' will wield a name with
        the following characteristics = YYYY-MM-DD-FIGURETITLE-my custom text

    - folder: set the folder to where the variable will be saved. By default
    creates and the folder 'Peridynamic variables' where the script is running and
    set it for saving the variables.
    Example: folder = r'C:\Users\USERNAME\Desktop\Peridynamic variables'
    - silent: wheter to print a message containg the full variable path of saved variable on success
    
    Output parameters:
    
    - saved file on disk'''

    if folder=='':
        folder=os.path.join(os.getcwd(),'Peridynamic variables')
    
    if os.path.isdir(folder)==False: # creates folder if it doesn't exist
        os.mkdir(folder)
    
    # Automattically set name based on know class objects
    var_names={pd.Mesh:'mesh',
    pd.Family:'family',
    pd.BoundaryConditions:'boundary conditions',
    pd.Model:'model',
    pd.general_functions.Energy:'energy'}

    try:
        name=name.replace('%v',var_names[obj.__class__]) # custom name formatting with plot attributes
    except KeyError:
        name=name.replace('%v','')

    name=strftime(name) # name formatting with time codes

    absolute_path=os.path.join(folder,name)

    with open(absolute_path,'wb') as file:
        pickle.dump(obj,file)

    if silent==False:
        print(f'Successfully saved file at {absolute_path}')

def load_object(absolute_path:str,silent:bool=False) -> None:
    '''Input parameters:
    
    - absolute_path: path to the file, includind the file and it's extension
    - silent: wheter to print a message containg the full variable path of loaded variable on success
    
    Output parameters:
    
    - loaded variable'''

    with open(absolute_path,'rb') as file:
        loaded_file=pickle.load(file)

    if silent==False:
        print(f'Successfully loaded file at {absolute_path}')

    return loaded_file
    