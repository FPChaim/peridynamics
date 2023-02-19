'''Provides the code necessary to generate a 2D peridynamic family'''

from sys import float_info
import numpy as np
import peridynamics as pd

def _IPA_AC(xk:float|int,yk:float|int,xi:float|int,yi:float|int,h:float|int,A:float|int,δ:float|int,counter:int,ε:float|int) -> tuple[float|int,float|int,float|int]:
    '''Returns partial area analyticial calculation used in "IPA-AC" algorith and the centroid coordinates as proposed by Seleson
    
    Input parameters:
    
    - xk: translated x from left to right
    - yk: translated y from bottom to top
    - xi: [xi,yi]
    - h: mesh grid spacing
    - A: elements' area
    - δ: peridynamic horizon
    - counter: number of corners of cell τk inside the neighborhood of i. Used to determine the case:
        - Case I: four corners of τk inside the neighborhood of i
        - Case II: three corners of τk inside the neighborhood of i 
        - Case III(a1): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk>yi
                - bottom-right corner of τk outside the neighborhood of i
        - Case III(a2): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk=yi
                - xk+h/2≥xi+δ
        - Case III(b): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk=yi
                - xk+h/2<xi+δ
        - Case III(c): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk>yi
                - bottom-right corner of τk inside the neighborhood of i
        - Case IV: one corner of τk inside the neighborhood of i
        - Case V: zero corners of τk inside the neighborhood of i
            - Condition:
                - xk-h/2<xi+δ

    Output parameters:

    - Ak: partial area
    - x̄k: centroid x coordinate
    - ȳk: centroid y coordinate'''

    #from numpy import sin,cos,np.arcsin,np.arccos,np.arctan

    if counter==4: # Case I

        # Partial area
        Ak=A # h²
        # Centroid
        x̄k=xk
        ȳk=yk

    elif counter==3: # Case II

        # Geometric parameters
        H1=yk+h/2-yi
        L1=(δ**2-H1**2)**0.5
        L2=xk+h/2-xi
        H2=(δ**2-L2**2)**0.5
        d=((H1-H2)**2+(L2-L1)**2)**0.5
        l=(δ**2-(d/2)**2)**0.5
        γ=np.arcsin(d/2/δ)
        # Partial area
        A1=h*(h-(H1-H2))
        A2=(h-(L2-L1))*(H1-H2)
        A3=1/2*(L2-L1)*(H1-H2)
        A4=γ*δ**2-1/2*d*l
        Ak=h**2-1/2*(L2-L1)*(H1-H2)+γ*δ**2-1/2*d*l
        # Centroid
        x̄1=xi+L2-h/2
        ȳ1=yi+H2-1/2*(h-(H1-H2))
        x̄2=xi+L1-1/2*(h-(L2-L1))
        ȳ2=yi+H1-1/2*(H1-H2)
        x̄3=xi+L2-2/3*(L2-L1)
        ȳ3=yi+H1-2/3*(H1-H2)
        θ=np.arctan(H2/L2)
        l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
        x̄4=xi+l̂*np.cos(θ+γ)
        ȳ4=yi+l̂*np.sin(θ+γ)
        x̄k=(A1*x̄1+A2*x̄2+A3*x̄3+A4*x̄4)/(A1+A2+A3+A4)
        ȳk=(A1*ȳ1+A2*ȳ2+A3*ȳ3+A4*ȳ4)/(A1+A2+A3+A4)

    elif counter==2: # Case III(a1), III(a2), III(b) or III(c)

        if yk>yi: # Case III(a1) or III(c)

            rbr_sq=(xk+h/2-xi)**2+(yk-h/2-yi)**2 # Distance squared of the bottom-right corner of τk from i
            
            if rbr_sq>δ**2: # bottom-right corner of τk is outside the neighborhood of i → Case III(a1)

                # Geometric parameters
                H1=yk+h/2-yi
                H2=yk-h/2-yi
                L1=(δ**2-H1**2)**0.5
                L2=(δ**2-H2**2)**0.5
                d=((L2-L1)**2+h**2)**0.5
                l=(δ**2-(d/2)**2)**0.5
                γ=np.arcsin(d/2/δ)
                # Partial area
                A1=(L1-(xk-h/2-xi))*h
                A2=1/2*(L2-L1)*h
                A3=γ*δ**2-1/2*d*l
                Ak=h*((L1+L2)/2-(xk-h/2-xi))+γ*δ**2-1/2*d*l
                # Centroid
                x̄1=xi+L1-1/2*(L1-(xk-h/2-xi))
                ȳ1=yi+H1-h/2
                x̄2=xi+L2-2/3*(L2-L1)
                ȳ2=yi+H1-2/3*h
                θ=np.arctan(H2/L2)
                l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
                x̄3=xi+l̂*np.cos(θ+γ)
                ȳ3=yi+l̂*np.sin(θ+γ)
                x̄k=(A1*x̄1+A2*x̄2+A3*x̄3)/(A1+A2+A3)
                ȳk=(A1*ȳ1+A2*ȳ2+A3*ȳ3)/(A1+A2+A3)

            else: # bottom-right corner of τk is inside the neighborhood of i → Case III(c)

                # Geometric parameters
                L1=xk-h/2-xi
                L2=xk+h/2-xi
                H1=(δ**2-L1**2)**0.5
                H2=(δ**2-L2**2)**0.5
                d=(h**2+(H1-H2)**2)**0.5
                l=(δ**2-(d/2)**2)**0.5
                γ=np.arcsin(d/2/δ)
                # Partial area
                A1=(H2-(yk-h/2-yi))*h
                A2=1/2*h*(H1-H2)
                A3=γ*δ**2-1/2*d*l
                Ak=h*((H1+H2)/2-(yk-h/2-yi))+γ*δ**2-1/2*d*l
                # Centroid
                x̄1=xi+L2-h/2
                ȳ1=yi+H2-1/2*(H2-(yk-h/2-yi))
                x̄2=xi+L1+1/3*h
                ȳ2=yi+H2+1/3*(H1-H2)
                l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
                θ=np.arctan(H2/L2)
                x̄3=xi+l̂*np.cos(θ+γ)
                ȳ3=yi+l̂*np.sin(θ+γ)
                x̄k=(A1*x̄1+A2*x̄2+A3*x̄3)/(A1+A2+A3)
                ȳk=(A1*ȳ1+A2*ȳ2+A3*ȳ3)/(A1+A2+A3)

        else: # Case III(a2) or III(b) | yk==yi

            if xk+h/2+ϵ>=xi+δ: # Case III(a2)

                # Geometric parameters
                l=(δ**2-(h/2)**2)**0.5
                γ=np.arcsin(h/2/δ)
                # Partial area
                A1=(l-(xk-h/2-xi))*h
                A2=γ*δ**2-1/2*h*l
                Ak=(l-(xk-h/2-xi))*h+(γ*δ**2-1/2*h*l)
                # Centroid
                x̄1=xi+l-1/2*(l-(xk-h/2-xi))
                ȳ1=yi
                l̂=4*δ*(np.sin(γ))**3/3/(2*γ-np.sin(2*γ))
                x̄2=xi+l̂
                ȳ2=yi
                x̄k=(A1*x̄1+A2*x̄2)/(A1+A2)
                ȳk=yi

            else: # Case III(b)

                # Geometric parameters
                l=(δ**2-(h/2)**2)**0.5
                L=xk+h/2-xi
                γ=np.arccos(l/δ)
                β=np.arccos(L/δ)
                d=2*(δ**2-L**2)**0.5
                # Partial area
                A1=(h-(L-l))*h
                A2=γ*δ**2-1/2*h*l
                A3=β*δ**2-1/2*d*L
                Ak=h**2-(L-l)*h+((γ*δ**2-1/2*h*l)-(β*δ**2-1/2*d*L))
                # Centroid
                x̄1=xi+l-1/2*(h-(L-l))
                ȳ1=yi
                l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
                x̄2=xi+l̂
                ȳ2=yi
                l̂_prime= 4*δ*np.sin(β)**3/3/(2*β-np.sin(2*β))
                x̄3=xi+l̂_prime
                ȳ3=yi
                x̄k=(A1*x̄1+A2*x̄2+A3*x̄3)/(A1+A2+A3)
                ȳk=yi

    elif counter==1: # Case IV

        # Geometric parameters
        L1=xk-h/2-xi
        H2=yk-h/2-yi
        H1=(δ**2-L1**2)**0.5
        L2=(δ**2-H2**2)**0.5
        d=((L2-L1)**2+(H1-H2)**2)**0.5
        l=(δ**2-(d/2)**2)**0.5
        γ=np.arcsin(d/2/δ)
        # Partial area
        A1=1/2*(L2-L1)*(H1-H2)
        A2=γ*δ**2-1/2*d*l
        Ak=1/2*(L2-L1)*(H1-H2)+γ*δ**2-1/2*d*l
        # Centroid
        x̄1=xi+L1+1/3*(L2-L1)
        ȳ1=yi+H2+1/3*(H1-H2)
        l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
        θ=np.arctan(H2/L2)
        x̄2=xi+l̂*np.cos(θ+γ)
        ȳ2=yi+l̂*np.sin(θ+γ)
        x̄k=(A1*x̄1+A2*x̄2)/(A1+A2)
        ȳk=(A1*ȳ1+A2*ȳ2)/(A1+A2)

    elif counter==0: # Case V

        if xk-h/2<xi+δ:

            # Geometric parameters
            l=xk-h/2-xi
            d=2*(δ**2-l**2)**0.5
            γ=np.arccos(l/δ)
            # Partial area
            Ak=γ*δ**2-1/2*d*l
            # Centroid
            l̂=4*δ*np.sin(γ)**3/3/(2*γ-np.sin(2*γ))
            x̄k=xi+l̂
            ȳk=yi

        else: # not needed because the vectors np.where initiated with np.zeros
            Ak=0
            # Centroid
            x̄k=0
            ȳk=0

    return Ak,x̄k,ȳk

def _PA_AC(xk:float|int,yk:float|int,xi:float|int,yi:float|int,h:float|int,A:float|int,δ:float|int,counter:int,ε:float|int) -> float|int:
    '''Returns partial area analyticial calculation used in "PA-AC" algorith as proposed by Seleson
    
    Input parameters:
    
    - xk: translated x from left to right
    - yk: translated y from bottom to top
    - xi: [xi,yi]
    - h: mesh grid spacing
    - A: elements' area
    - δ: peridynamic horizon
    - case: algorithm case as proposed by Seleson | type: str
        - I: four corners of τk inside the neighborhood of i
        - II: three corners of τk inside the neighborhood of i
        - III(a1): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk>yi
                - bottom-right corner of τk outside the neighborhood of i
        - III(a2): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk=yi
                - xk+h/2≥xi+δ
        - III(b): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk=yi
                - xk+h/2<xi+δ
        - III(c): two corners of τk inside the neighborhood of i
            - Conditions:
                - yk>yi
                - bottom-right corner of τk inside the neighborhood of i
        - IV: one corner of τk inside the neighborhood of i
        - V: zero corners of τk inside the neighborhood of i
            - Condition:
                - xk-h/2<xi+δ

    Output parameters:

    - Ak: partial area'''

    #from numpy import np.arcsin,np.arccos

    if counter==4: # Case I

        # Partial area
        Ak=A # h²

    elif counter==3: # Case II

        # Geometric parameters
        H1=yk+h/2-yi
        L1=(δ**2-H1**2)**0.5
        L2=xk+h/2-xi
        H2=(δ**2-L2**2)**0.5
        d=((H1-H2)**2+(L2-L1)**2)**0.5
        l=(δ**2-(d/2)**2)**0.5
        γ=np.arcsin(d/2/δ)
        # Partial area
        Ak=h**2-1/2*(L2-L1)*(H1-H2)+γ*δ**2-1/2*d*l

    elif counter==2: # Case III(a1), III(a2), III(b) or III(c)

        if yk>yi: # Case III(a1) or III(c)

            rbr_sq=(xk+h/2-xi)**2+(yk-h/2-yi)**2 # Distance squared of the bottom-right corner of τk from i

            if rbr_sq>δ**2: # bottom-right corner of τk is outside the neighborhood of i → Case III(a1)

                # Geometric parameters
                H1=yk+h/2-yi
                H2=yk-h/2-yi
                L1=(δ**2-H1**2)**0.5
                L2=(δ**2-H2**2)**0.5
                d=((L2-L1)**2+h**2)**0.5
                l=(δ**2-(d/2)**2)**0.5
                γ=np.arcsin(d/2/δ)
                # Partial area
                Ak=h*((L1+L2)/2-(xk-h/2-xi))+γ*δ**2-1/2*d*l

            else: # bottom-right corner of τk is inside the neighborhood of i → Case III(c)

                # Geometric parameters
                L1=xk-h/2-xi
                L2=xk+h/2-xi
                H1=(δ**2-L1**2)**0.5
                H2=(δ**2-L2**2)**0.5
                d=(h**2+(H1-H2)**2)**0.5
                l=(δ**2-(d/2)**2)**0.5
                γ=np.arcsin(d/2/δ)
                # Partial area
                Ak=h*((H1+H2)/2-(yk-h/2-yi))+γ*δ**2-1/2*d*l

        else: # Case III(a2) or III(b) | yk==yi

            if xk+h/2+ϵ>=xi+δ: # Case III(a2)

                # Geometric parameters
                l=(δ**2-(h/2)**2)**0.5
                γ=np.arcsin(h/2/δ)
                # Partial area
                Ak=(l-(xk-h/2-xi))*h+(γ*δ**2-1/2*h*l)

            else: # Case III(b)

                # Geometric parameters
                l=(δ**2-(h/2)**2)**0.5
                L=xk+h/2-xi
                γ=np.arccos(l/δ)
                β=np.arccos(L/δ)
                d=2*(δ**2-L**2)**0.5
                # Partial area
                Ak=h**2-(L-l)*h+((γ*δ**2-1/2*h*l)-(β*δ**2-1/2*d*L))

    elif counter==1: # Case IV

        # Geometric parameters
        L1=xk-h/2-xi
        H2=yk-h/2-yi
        H1=(δ**2-L1**2)**0.5
        L2=(δ**2-H2**2)**0.5
        d=((L2-L1)**2+(H1-H2)**2)**0.5
        l=(δ**2-(d/2)**2)**0.5
        γ=np.arcsin(d/2/δ)
        # Partial area
        Ak=1/2*(L2-L1)*(H1-H2)+γ*δ**2-1/2*d*l

    elif counter==0: # Case V

        if xk-h/2<xi+δ:

            # Geometric parameters
            l=xk-h/2-xi
            d=2*(δ**2-l**2)**0.5
            γ=np.arccos(l/δ)
            # Partial area
            Ak=γ*δ**2-1/2*d*l

        else:  # not needed because the vectors np.where initiated with np.zeros
            Ak=0
    
    return Ak

class Family:
    '''Peridynamic family object, with the following attributes:

        - j: vector containing all neighbors j for all points i
        - maxNeigh: the maximum number of neighbors around a node (scalar)
        - PA_alg: partial area algorithm used
        - PAj: partial areas of points j for each point i
        - SC_alg: surface correction algorithm usedd
        - SCj: surface correction of points j for each point i
        - xj: position of points j for each point i
        - XJ: position with the corrected quadrature coordinates x and y for j-th node at the neighborhood of the i-th node'''

    def __init__(self,mesh:pd.Mesh,δ:float|int,m:float|int,PA_alg:str,SC_alg:str,silent:bool=False) -> None:
        '''A class that one should intializate as an object, e.g: family=Family()
        
        Input parameters:
        
        - mesh: peridynamic mesh. The following mesh parameters will be used in this function:
            - h: grid spacing | type: float or int
            - x: Position of the nodes
            - A: Areas vector (or scalar)
        - δ: Peridynamic horizon
        - m: mesh ratio (m=δ/h), which represents the number of nodes inside the horizon
        - PA_alg: Partial area algorithm
            - "IPA-AC": improved partial area analytical calcultation algorithm proposed by Seleson (partially implemented - the centroids are calculated but aren't used)
            - "PA-AC": partial area analytical calcultation algorithm proposed by Seleson (recommended)
            - "PA-HHB+": vectorized PA-HHB
            - "PA-HHB": partial area algorithm proposed by Hu, Ha and Bobaru
            - "PA-PDLAMMPS+": vectorized PA-PDLAMMPS
            - "PA-PDLAMMPS": Partial area algorithm used in the PDLAMMPS software
            - "FA+": vectorized FA
            - "FA": full area algorithm proposed by Silling and Askari
        - SC_alg: surface correction algorithm
            - "Volume": volume correction
            - "None": no correction
        - silent: wheter to print feedback during calculation
        
        Output parameters:
        
            Updates the following attributes:

            - j: vector containing all neighbors j for all points i
            - maxNeigh: the maximum number of neighbors around a node (scalar)
            - PA_alg: partial area algorithm used
            - PAj: partial areas of points j for each point i
            - SC_alg: surface correction algorithm usedd
            - SCj: surface correction of points j for each point i
            - xj: position of points j for each point i
            - XJ: position with the corrected quadrature coordinates x and y for j-th node at the neighborhood of the i-th node'''

        self.PA_alg=PA_alg
        self.SC_alg=SC_alg

        # Renaming variables to make the code more readable
        h=mesh.h
        x=mesh.points
        A=mesh.A
        
        # Start calculations
        max_columns=int((2*np.ceil(m)+1)**2) # maximum possible number of neighbors for np.any point
        ϵ=float_info.epsilon # very small number of the order of the machine precision
        maxNeigh=0
        family=[] # indices (j) of neighbors of point xi
        # i_and_family=[]
        xj=[] # values of points x[j]
        # xi_and_xj=[]
        PAj=[] # partial areas of cells related to points xj
        # PAi_and_PAj=[]
        XJ=[]
        # XI_and_XJ=[]
        # YJ=[]
        # YI_and_YJ=[]

        if PA_alg=='IPA-AC':

            # Algorithm 1: compute interactions
            # - i: index of all the mesh points
            # - col_i: len(x)/ncols loops of columns indexes (0,1,2,...,ncols-1,0,1,2,...,ncols-1,...) <-- column of current point
            # - row_i: (0,0,0,...,ncols-1,1,1,1,...,ncols-1,...) <-- row of curent point
            # - k: every point around point i that can be a neighbor ?
            
            numx=mesh.ncols
            numy=mesh.nrows

            # Compute the maximum number of one-sided neighbors interactions on the x-direction
            Nx=int(np.floor(δ/h+1/2-ϵ))
        
            for i,col_i in zip(range(0,len(x)),[_ for _ in range(0,numx)]*(int(len(x)/numx))):
                
                # Initialize the family vector for the point i
                family_i=np.full((max_columns),-1,dtype=int)
                number_of_neighbors_of_i=0 # Number of points whithin the neighbourhood of i
                
                if col_i<Nx:
                    Nx_esq=col_i
                    Nx_dir=Nx
                elif numx-(col_i+1)<Nx:
                    Nx_esq=Nx
                    Nx_dir=numx-(col_i+1)
                else:
                    Nx_esq=Nx
                    Nx_dir=Nx
                row_i=int(np.ceil((i+1)/numx)-1) # x = (xiI,y_j); first line, row_i = 0; last line, row_i = numx-1      
                for k in range(col_i-Nx_esq,col_i+Nx_dir+1):                                
                    # Compute maximum number of one-sided neighbor interaction along the y-direction Ny
                    if k==col_i:
                        Ny=Nx
                    else:
                        ξ1=abs(x[i,0]-x[k+(row_i-1)*numx,0])
                        Ny=int(np.floor((δ**2-(ξ1-h/2)**2)**0.5/h+1/2-ϵ))
                    # Check for borders
                    if row_i<Ny:
                        Ny_bot=row_i
                        if numy-(row_i+1)<Ny:
                            Ny_top=numy-(row_i+1)
                        else:
                            Ny_top=Ny
                    elif numy-(row_i+1)<Ny:
                        Ny_bot=Ny
                        Ny_top=numy-(row_i+1)
                    else:
                        Ny_bot=Ny
                        Ny_top=Ny
                    
                    for l in range(row_i-Ny_bot,row_i+Ny_top+1):
                        if l*numx+k!=i:
                            family_i[number_of_neighbors_of_i]=l*numx+k
                            number_of_neighbors_of_i=number_of_neighbors_of_i+1
                
                #family_i=family_i[family_i>-1]
                family_i=family_i[:number_of_neighbors_of_i]
                family.append(family_i) # appends valids neighbors of point xi

                xj_i=x[family_i]
                xj.append(xj_i)
                
                if number_of_neighbors_of_i>maxNeigh:
                    maxNeigh=number_of_neighbors_of_i
        
                # Algorithm 2: compute partial areas
                # Map neighbor cell to top-right quadrant
                
                # Initialize the partial areas vector
                PAj_i=np.zeros((len(family_i)),dtype=float)

                # Initialize the centroid vector
                XJ_i=np.zeros((len(family_i)),dtype=float)
                YJ_i=np.zeros((len(family_i)),dtype=float)

                # Translate points: (xj,yj)[i] → (xk,yk)[i]
                # xj_i[neigh_index][0]: xj[i]
                # xj_i[neigh_index][1]: yj[i]
                for neigh_index in range(0,number_of_neighbors_of_i):

                    # Initialize moving flags
                    f_lr=False # left to right
                    f_bt=False # bottom to top
                    f_rot=False # rotation from +y-axis to +x-axis

                    if xj_i[neigh_index,0]<x[i,0]: # compare x component of the expression xj<xj_i[neigh_index][0]
                        xk=x[i,0]+(x[i,0]-xj_i[neigh_index,0]) # Translate cell from left to right
                        f_lr=True
                    else:
                        xk=xj_i[neigh_index,0]
                    if xj_i[neigh_index,1]<x[i,1]:
                        yk=x[i,1]+(x[i,1]-xj_i[neigh_index,1]) # Translate cell from bottom to top
                        f_bt=True
                    else:
                        yk=xj_i[neigh_index,1]
                    if xj_i[neigh_index,0]==x[i,0]:
                        # Rotate cell from +y-axis to +x-axis
                        Delta=yk-x[i,1]
                        yk=x[i,1]
                        xk=x[i,0]+Delta
                        f_rot=True

                    # Count number of corners of cells τk inside the neighborhood of each point xj_i[neigh_index][0]
                    counter=0
                    for n in [-1,1]:
                        for m in [-1,1]:
                            if (xk+n*h/2-x[i,0])**2+(yk+m*h/2-x[i,1])**2<δ**2:
                                counter=counter+1

                    # # Count number of corners of cells τk inside the neighborhood of each point xi
                    # n=np.array([[-1],[1]]) # (2,1)
                    # m=np.array([-1,1]) # (2) --> n*m --> (2,2)
                    # # m=np.array([[-1],[1]])
                    # counter=np.sum((xk+n*h/2-x[i,0])**2+(yk+m*h/2-x[i,1])**2<δ**2)

                    # Compute partial area and centroid
                    Ak,x̄k,ȳk=_IPA_AC(xk,yk,x[i,0],x[i,1],h,A,δ,counter,ε)

                    # Map centroid back to the original quadrant of τk
                    if f_rot==True:
                        # Rotate centroid from +x-axis to +y-axis
                        Delta=x̄k-x[i,0]
                        x̄k=x[i,0]
                        ȳk=x[i,1]+Delta

                    if f_bt==True:
                        # Translate centroid from top to bottom
                        ȳk=x[i,1]-(ȳk-x[i,1])

                    if f_lr==True:
                        # Translate centroid from right to left
                        x̄k=x[i,0]-(x̄k-x[i,0])

                    PAj_i[neigh_index]=Ak
                    XJ_i[neigh_index]=x̄k
                    YJ_i[neigh_index]=ȳk

                PAj.append(PAj_i)
                XJ.append(np.column_stack((XJ_i,YJ_i)))
                #YJ.append(YJ_i)

                if silent==False:            
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')

        elif PA_alg=='PA-AC':

            # Algorithm 1: compute interactions
            # - i: index of all the mesh points
            # - col_i: len(x)/ncols loops of columns indexes (0,1,2,...,ncols-1,0,1,2,...,ncols-1,...) <-- column of current point
            # - row_i: (0,0,0,...,ncols-1,1,1,1,...,ncols-1,...) <-- row of curent point
            # - k: every point around point i that can be a neighbor ?
            
            numx=mesh.ncols
            numy=mesh.nrows

            # Compute the maximum number of one-sided neighbors interactions on the x-direction
            Nx=int(np.floor(δ/h+1/2-ϵ))
        
            for i,col_i in zip(range(0,len(x)),[_ for _ in range(0,numx)]*(int(len(x)/numx))):
                
                # Initialize the family vector for the point i
                family_i=np.full((max_columns),-1,dtype=int)
                number_of_neighbors_of_i=0 # Number of points whithin the neighbourhood of i
                
                if col_i<Nx:
                    Nx_esq=col_i
                    Nx_dir=Nx
                elif numx-(col_i+1)<Nx:
                    Nx_esq=Nx
                    Nx_dir=numx-(col_i+1)
                else:
                    Nx_esq=Nx
                    Nx_dir=Nx
                row_i=int(np.ceil((i+1)/numx)-1) # x = (xiI,y_j); first line, row_i = 0; last line, row_i = numx-1      
                for k in range(col_i-Nx_esq,col_i+Nx_dir+1):                                
                    # Compute maximum number of one-sided neighbor interaction along the y-direction Ny
                    if k==col_i:
                        Ny=Nx
                    else:
                        ξ1=abs(x[i,0]-x[k+(row_i-1)*numx,0])
                        try:
                            Ny=int(np.floor((δ**2-(ξ1-h/2)**2)**0.5/h+1/2-ϵ))
                        except ValueError:
                            Ny=0
                    # Check for borders
                    if row_i<Ny:
                        Ny_bot=row_i
                        if numy-(row_i+1)<Ny:
                            Ny_top=numy-(row_i+1)
                        else:
                            Ny_top=Ny
                    elif numy-(row_i+1)<Ny:
                        Ny_bot=Ny
                        Ny_top=numy-(row_i+1)
                    else:
                        Ny_bot=Ny
                        Ny_top=Ny
                    
                    for l in range(row_i-Ny_bot,row_i+Ny_top+1):
                        if l*numx+k!=i: # skip the point xi
                            family_i[number_of_neighbors_of_i]=l*numx+k
                            number_of_neighbors_of_i=number_of_neighbors_of_i+1
                
                #family_i=family_i[family_i>-1]
                family_i=family_i[:number_of_neighbors_of_i]
                family.append(family_i) # appends valids neighbors of point xi

                xj_i=x[family_i]
                xj.append(xj_i)
                
                if number_of_neighbors_of_i>maxNeigh:
                    maxNeigh=number_of_neighbors_of_i
        
                # Algorithm 2: compute partial areas
                # Map neighbor cell to top-right quadrant
                
                # Initialize the partial areas vector
                PAj_i=np.zeros((len(family_i)),dtype=float)

                # Translate points: (xj,yj)[i] → (xk,yk)[i]
                # xj_i[neigh_index][0]: xj[i]
                # xj_i[neigh_index][1]: yj[i]
                for neigh_index in range(0,number_of_neighbors_of_i):
                    if xj_i[neigh_index,0]<x[i,0]: # compare x component of the expression xj<xi
                        xk=x[i,0]+(x[i,0]-xj_i[neigh_index,0]) # Translate cell from left to right
                    else:
                        xk=xj_i[neigh_index,0]
                    if xj_i[neigh_index,1]<x[i,1]:
                        yk=x[i,1]+(x[i,1]-xj_i[neigh_index,1]) # Translate cell from bottom to top
                    else:
                        yk=xj_i[neigh_index,1]
                    if xj_i[neigh_index,0]==x[i,0]:
                        # Rotate cell from +y-axis to +x-axis
                        Delta=yk-x[i,1]
                        yk=x[i,1]
                        xk=x[i,0]+Delta

                    # # Count number of corners of cells τk inside the neighborhood of each point xi
                    # counter=0
                    # counter_i=0
                    # for n in [-1,1]:
                    #     for m in [-1,1]:
                    #         if (xk+n*h/2-x[i,0])**2+(yk+m*h/2-x[i,1])**2<δ**2:
                    #             counter=counter+1
                    #         # Count number of corners of cell τi (to include itself in PAi_and_PAj)
                    #         if (n*h/2)**2+(m*h/2)**2<δ**2:
                    #             counter_i=counter_i+1

                    # Count number of corners of cells τk inside the neighborhood of each point xi
                    n=np.array([[-1],[1]]) # (2,1)
                    m=np.array([-1,1]) # (2) --> n*m --> (2,2)
                    counter=np.sum((xk+n*h/2-x[i,0])**2+(yk+m*h/2-x[i,1])**2<δ**2)
                    #counter_i=np.sum((n*h/2)**2+(m*h/2)**2<δ**2) # counter os xi

                    # Compute partial area
                    Ak=_PA_AC(xk,yk,x[i,0],x[i,1],h,A,δ,counter,ε)
                    PAj_i[neigh_index]=Ak

                PAj.append(PAj_i)

                if silent==False:            
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')

        elif PA_alg=='PA-HHB+':

            # Fi(HHB) := {k : xk ∈ LB and |xk − xi| − h2 ≤ δ}
            # i represents the indice of the points x[i] = [xi,yi] - reference points
            # k represents the indice of the points x[k] = [xk,yk] - neighborhood candidate points
            
            for i in range(0,len(x)): # loops through all points

                # Calculate conditions, get indices and generate the family vector for the point xi
                ξ=np.linalg.norm(x[i]-x,axis=1)
                k1=np.where(np.logical_and(ξ+h/2<=δ+ϵ,ξ>0))[0] # ξ>0 remove distance from own point
                j1=len(k1) # number of neighbors of the point xi for condition 1

                k2=np.where(np.logical_and(ξ-h/2<=δ+ϵ,ξ>0))[0] # ξ>0 remove distance from own point
                k2=np.setdiff1d(k2,k1,assume_unique=True) # this algorithm allows both conditions to return the same indices, so we only keep new indices
                j2=len(k2) # number of neighbors of the point xi for condition 2
                
                # Initialize the family vector for the point xi
                # The order is a little bit different, but the sorted np.array is the same as 'PA-HHB'
                family_i=np.empty((j1+j2),dtype=int)
                family_i[:j1]=k1
                family_i[j1:j1+j2]=k2 # the order is a little bit different, but the sorted np.array is the same as 'PA-HHB'

                xj_i=x[family_i]
                xj.append(xj_i)

                # Check for maxNeigh
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)

                # Calculate partial areas and finish up
                PAj_i=np.empty((len(family_i)),dtype=float)
                PAj_i[:j1]=A
                PAj_i[j1:j1+j2]=1/h*(δ-(ξ[k2]-h/2))*A # A=h²
                family.append(family_i)
                PAj.append(PAj_i)
                
                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')

        elif PA_alg=='PA-HHB':

            # Fi(HHB) := {k : xk ∈ LB and |xk − xi| − h2 ≤ δ}
            # i represents the indice of the points x[i] = [xi,yi] - reference points
            # k represents the indice of the points x[k] = [xk,yk] - neighborhood candidate points
            
            for i in range(0,len(x)): # loops through all points

                # Initialize the family and partial area vectors for the point xi
                family_i=np.empty((max_columns),dtype=int)
                PAj_i=np.empty((max_columns),dtype=float)
                j=0 # initiates the column index used for the family and partial area vectors
                
                # Loops through possible neighbors of points xi, checks condition and generate vectors
                for k in range(0,len(x)): # loops through all point again (possible neighbors)
                    if i!=k: # only execute if xi!=xk (ignore checking if xi is neighbor of itself)
                        # Compute reference distance between cell centers
                        ξ=np.linalg.norm(x[k]-x[i])
                        # Check if cell τk is contained within the neighborhood of xi
                        if ξ+h/2<=δ+ϵ:
                            family_i[j]=k
                            PAj_i[j]=A # A=h²
                            j=j+1
                        elif ξ-h/2<=δ+ϵ:
                            family_i[j]=k
                            PAj_i[j]=1/h*(δ-(ξ-h/2))*A # A=h²
                            j=j+1
                        # else: # not needed because we initiated the PA vector with np.zeros
                            # PAj_i[j]=0
                
                # Clean up both lists for the point xi, check for maxNeigh and finish up
                family_i=family_i[:j]

                xj_i=x[family_i]
                xj.append(xj_i)

                PAj_i=PAj_i[:j]
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)
                family.append(family_i)
                PAj.append(PAj_i)
                        
                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
        
        elif PA_alg=='PA-PDLAMMPS+':

            # Fi(FA) = Fi(PA-PDLAMMPS) := {k : xk ∈ LB and |xk − xi| ≤ δ}
            # i represents the indice of the points x[i] = [xi,yi] - reference points
            # k represents the indice of the points x[k] = [xk,yk] - neighborhood candidate points
            
            for i in range(0,len(x)): # loops through all points

                # Calculate conditions, get indices and generate the family vector for the point xi
                ξ=np.linalg.norm(x[i]-x,axis=1)
                k1=np.where(np.logical_and(ξ+h/2<=δ+ϵ,ξ>0))[0] # ξ>0 remove distance from own point
                j1=len(k1) # number of neighbors of the point xi for condition 1
                
                k2=np.where(np.logical_and(ξ<=δ+ϵ,ξ>0))[0] # ξ>0 remove distance from own point
                k2=np.setdiff1d(k2,k1,assume_unique=True) # this algorithm allows both conditions to return the same indices, so we only keep new indices
                j2=len(k2) # number of neighbors of the point xi for condition 2
                
                # Initialize the family vector for the point xi
                family_i=np.empty((j1+j2),dtype=int)
                family_i[:j1]=k1
                family_i[j1:j1+j2]=k2 # the order is a little bit different, but the sorted np.array is the same as 'PA-HHB'

                xj_i=x[family_i]
                xj.append(xj_i)

                # Check for maxNeigh
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)

                # Calculate partial areas and finish up
                PAj_i=np.empty((len(family_i)),dtype=float)
                PAj_i[:j1]=A
                PAj_i[j1:j1+j2]=1/h*(δ-(ξ[k2]-h/2))*A # A=h²
                family.append(family_i)
                PAj.append(PAj_i)
                
                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
    
        elif PA_alg=='PA-PDLAMMPS':

            # Fi(FA) = Fi(PA-PDLAMMPS) := {k : xk ∈ LB and |xk − xi| ≤ δ}
            # i represents the indice of the points x[i] = [xi,yi] - reference points
            # k represents the indice of the points x[k] = [xk,yk] - neighborhood candidate points
            
            for i in range(0,len(x)): # loops through all points
                
                # Initialize the family and partial area vectors for the point xi
                family_i=np.empty((max_columns),dtype=int)
                PAj_i=np.empty((max_columns),dtype=float)
                j=0 # initiates the column index used for the family and partial area vectors

                # Loops through possible neighbors of points xi, checks condition and generate vectors
                for k in range(0,len(x)): # loops through all point again (possible neighbors)
                    if i!=k: # only execute if xi!=xk (ignore checking if xi is neighbor of itself)
                        # Compute reference distance between cell centers
                        ξ=np.linalg.norm(x[k]-x[i])
                        # Check if cell τk is contained within the neighborhood of i
                        if ξ+h/2<=δ+ϵ:
                            family_i[j]=k
                            PAj_i[j]=A # A=h²
                            j=j+1
                        elif ξ<=δ+ϵ:
                            family_i[j]=k
                            PAj_i[j]=1/h*(δ-(ξ-h/2))*A # A=h²
                            j=j+1
                        # else: # not needed because we initiated the PA vector with np.zeros
                            # partialAreas[i,j]=0
                
                # Clean up both lists for the point xi, check for maxNeigh and finish up
                family_i=family_i[:j]

                xj_i=x[family_i]
                xj.append(xj_i)

                PAj_i=PAj_i[:j]
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)
                family.append(family_i)
                PAj.append(PAj_i)

                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')

        elif PA_alg=='FA+':

            # Fi(FA) = Fi(PA-PDLAMMPS) := {k : xk ∈ LB and |xk − xi| ≤ δ}
            # i represents the indice of the points [xi,yi] - reference points
            # k represents the indice of the points [xk,yk] - neighborhood candidate points

            for i in range(0,len(x)): # loops through all points

                # Get family of point xi directly from  the condition
                ξ=np.linalg.norm(x[i]-x,axis=1) # distance from point xi to all the mesh points (includind self)
                family_i=np.where(np.logical_and(ξ<=δ+ϵ,ξ>0))[0] # ξ>0 remove distance from own point

                xj_i=x[family_i]
                xj.append(xj_i)

                # Check for maxNeigh
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)

                # Calculate partial areas and finish up
                PAj_i=np.full(len(family_i),A,dtype=float)
                family.append(family_i)
                PAj.append(PAj_i)
                        
                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')                
                    
        elif PA_alg=='FA':

            # Fi(FA) = Fi(PA-PDLAMMPS) := {k : xk ∈ LB and |xk − xi| ≤ δ}.
            # i represents the indice of the points [xi,yi] - reference points
            # k represents the indice of the points [xk,yk] - neighborhood candidate points

            for i in range(0,len(x)): # loops through all points

                # Initialize the family and partial area vectors for the point xi
                family_i=np.empty((max_columns),dtype=int)
                PAj_i=np.empty((max_columns),dtype=float)
                j=0 # initiates the column index used for the family and partial area vectors

                # Loops through possible neighbors of points xi, checks condition and generate vectors
                for k in range(0,len(x)): # loops through all points again
                    if i!=k: # only execute if xi!=xk (ignore checking if xi is neighbor of itself)
                        # Compute reference distance between cell centers
                        ξ=np.linalg.norm(x[k]-x[i]) # distance from point xi to all the mesh points (includind self)
                        # Check if point k is in the family of i
                        if ξ<=δ+ϵ:
                            family_i[j]=k
                            PAj_i[j]=A # A=h²
                            j=j+1
                        # else: # not needed because we initiated the PA vector with np.zeros
                            # partialAreas[i,j]=0

                # Clean up both lists for the point xi, check for maxNeigh and finish up
                family_i=family_i[:j]

                xj_i=x[family_i]
                xj.append(xj_i)

                PAj_i=PAj_i[:j]
                if len(family_i)>maxNeigh:
                    maxNeigh=len(family_i)
                family.append(family_i)
                PAj.append(PAj_i)

                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating family and partial areas... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
        
        else:
            raise NameError('The available Partial Area algorithms are: "IPA-AC", "PA-AC" (recommended), "PA-HHB+", "PA-HHB", "PA-PDLAMMPS+","PA-PDLAMMPS","FA+" and "FA"')

        family=np.array(family,dtype=object)
        xj=np.array(xj,dtype=object)
        PAj=np.array(PAj,dtype=object)
        XJ=np.array(XJ,dtype=object)
        #YJ=np.array(YJ,dtype=object)

        # Surface Correction algorithms
        # surfaceCorrection=np.ones((len(PAj)),dtype=float)
        # if SC_alg=='Volume+':
        #     surfaceCorrection=[]
        #     surfaceCorrection=np.ones((len(PAj)),dtype=float)
        #     V0=pi*δ**2 # V0=pi*δ**2 for 2D | V0=4/3*pi*δ**3 for 3D
        #     #for i in range(0,len(x)):
        #     for i in range(0,len(PAj[i])):
        #         j=family[i]>-1 # valid family columns
        #         surfaceCorrection[i,j]=2*V0/(np.sum(partialAreas[i])+np.sum(partialAreas[family[i,j]],axis=1))
        #         if silent==False:
        #             pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating surface correction... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
        if SC_alg=='Volume':
            # surfaceCorrection=np.ones((len(PAj)),dtype=float)
            surfaceCorrection=[]
            V0=np.pi*δ**2 # V0=pi*δ**2 for 2D | V0=4/3*pi*δ**3 for 3D
            for i in range(0,len(x)):
                surfaceCorrection_i=np.ones((len(PAj[i])),dtype=float)
                np.sum_PAj_i=np.sum(PAj[i])
                family_i=family[i]
                for j in range(0,len(PAj[i])):
                    surfaceCorrection_i[j]=2*V0/(np.sum_PAj_i+np.sum(PAj[family_i[j]]))
                surfaceCorrection.append(surfaceCorrection_i)
                if silent==False:
                    pd._misc.progress_printer(values=(i+1)/len(x),fmt='Calculating surface correction... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
            surfaceCorrection=np.array(surfaceCorrection,dtype=object)
        elif SC_alg=='None':
            #surfaceCorrection=1.
            surfaceCorrection=np.ones((len(PAj)),dtype=float)
            if silent==False:
                pd._misc.progress_printer(values=1,fmt='Calculating surface correction... |%b| %p %',bar_sep=40,hundred_percent_print=' Done.')
        else:
            raise NameError('The available Surface Correction algorithms are: "Volume" (recommended) and "None" (for no correction)')

        self.j=family
        self.xj=xj
        self.maxNeigh=maxNeigh
        self.PAj=PAj
        self.XJ=XJ
        #self.YJ=YJ
        self.SCj=surfaceCorrection

        # if silent==False:
        #     print('Done.')
