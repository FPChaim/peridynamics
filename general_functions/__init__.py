'''Some scatter functions that are used somewhere else'''

from peridynamics.general_functions.general_functions_boundary_conditions import Damage
from peridynamics.general_functions.general_functions_model import History,influenceFunction,weightedVolume
from peridynamics.general_functions.general_functions_solvers import checkBondCrack,Energy,fload,getForce,interp1d,parFor_loop,sampling,tangentStiffnessMatrix
