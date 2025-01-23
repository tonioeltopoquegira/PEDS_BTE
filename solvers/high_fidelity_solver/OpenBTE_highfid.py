# High fidelity solver using the open-source package OpenBTE.
# OpenBTE, an efficient solver for the steady-state phonon BTE in multidimensional structures.
# This tool is interfaced to first-principles calculations, thus it unlocks the calculations of thermal-related properties with no fitting-parameters

import numpy as np
import matplotlib.pyplot as plt

# OpenBTE
from openbte import load_rta, RTA2DSym, Geometry, BTE_RTA, Fourier, rectangle, get_mesh
from openbte.objects import BoundaryConditions, OpenBTEResults, EffectiveThermalConductivity


def highfidelity_solver(pores, save_show_res = False):

    print(pores)

    # cancel any previous geometry saved 

    # Create Material
    rta_data = load_rta('Si_rta') 
    mat = RTA2DSym(data=rta_data)

    # Create Base Mesh
    mesh = Geometry(2) # prova anche 1
    L= 100 # mesh 
    mesh.add_shape(rectangle(area = L*L))

    # Pores
    pores_centers = convert_pores(pores)
    for pore in pores_centers:
        mesh.add_hole(rectangle(area = 100,x=pore[0],y=pore[1]))

    # Set Boundary conditions
    if pores.any():
        mesh.set_boundary_region(selector = 'inner',region = 'Boundary')

    mesh.set_periodicity(direction = 'x',region = 'Periodic_x')
    mesh.set_periodicity(direction = 'y',region = 'Periodic_y')

    
    mesh.save() 

    mesh = get_mesh()

    # Boundary Conditions
    if pores.any():
        boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 0,'Periodic_y':1}, diffuse='Boundary')
    
    else:
        boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 0,'Periodic_y':1})
    # Effective Thermal Conductivity
    effective_kappa = EffectiveThermalConductivity(normalization=-1,contact='Periodic_y')

    #print(mat.thermal_conductivity)

    # Base Solver for standard Heat Conduction (first guess)
    fourier = Fourier(mesh,mat.thermal_conductivity,boundary_conditions, effective_thermal_conductivity=effective_kappa, verbose=False)
    

    # Boltzmann Transport EquationSolver
    bte = BTE_RTA(mesh ,mat ,boundary_conditions ,fourier=fourier, effective_thermal_conductivity=effective_kappa)

    results = OpenBTEResults(mesh=mesh,material = mat,solvers={'bte':bte})
    
    results.save()
    results.show()

    results = OpenBTEResults.load()

    
    results_bte = results[-2]['bte']
    #results_fourier = results[-2]['fourier']

    kappa_eff_BTE, temp_BTE, flux_BTE = results_bte.kappa_eff, results_bte.variables['Temperature_BTE']['data'], results_bte.variables['Flux_BTE']['data']
    #kappa_eff_Fourier, temp_Fourier, flux_Fourier = results_fourier.kappa_eff, results_fourier.variables['Temperature_Fourier'], results_fourier.variables['Flux_Fourier']

    # Possibly need to integrate the flux of Fourier 

    return kappa_eff_BTE, temp_BTE, flux_BTE


def convert_pores(pores):
    
    indices = np.argwhere(pores)

    indices = indices[:, [1, 0]]  

    # Calculate the centers for each pore
    pores_centers = (indices * (-20) +40).tolist()

    for p in pores_centers:
        p[0] = -p[0]
    
    return pores_centers


if  __name__ == "__main__":

    #pores = np.random.rand(5, 5) < 0.5

    pores = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])

    pores = pores.reshape((5,5))
    
    results = highfidelity_solver(pores, save_show_res=True)

    kappa, _, _ = results

    print(kappa)