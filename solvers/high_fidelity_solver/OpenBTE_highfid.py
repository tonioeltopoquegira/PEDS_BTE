# High fidelity solver using the open-source package OpenBTE.
# OpenBTE, an efficient solver for the steady-state phonon BTE in multidimensional structures.
# This tool is interfaced to first-principles calculations, thus it unlocks the calculations of thermal-related properties with no fitting-parameters

import numpy as np
import os

# OpenBTE
from openbte import load_rta, RTA2DSym, Geometry, BTE_RTA, Fourier, rectangle, get_mesh
from openbte.objects import BoundaryConditions, OpenBTEResults, EffectiveThermalConductivity


def highfidelity_solver(pores, step_size, save_show_res = False):

    # cancel any previous geometry saved 

    # Create Material
    rta_data = load_rta('Si_rta') 
    mat = RTA2DSym(data=rta_data)

    # Create Base Mesh
    mesh = Geometry(step_size) # prova anche 1
    L= 100 # mesh 
    mesh.add_shape(rectangle(area = L*L))

    # Pores
    pores_info = convert_pores(pores)
    '''for area, x, y in pores_info:
        mesh.add_hole(rectangle(area=area, x=x, y=y))'''
    for x, y in pores_info:
        mesh.add_hole(rectangle(area=100, x=x, y=y))

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



    results = OpenBTEResults.load()

    
    results_bte = results[-2]['bte']
    #results_fourier = results[-2]['fourier']

    kappa_eff_BTE, temp_BTE, flux_BTE = results_bte.kappa_eff, results_bte.variables['Temperature_BTE']['data'], results_bte.variables['Flux_BTE']['data']
    #kappa_eff_Fourier, temp_Fourier, flux_Fourier = results_fourier.kappa_eff, results_fourier.variables['Temperature_Fourier'], results_fourier.variables['Flux_Fourier']

    # Possibly need to integrate the flux of Fourier 

    
    results.show(include=["Temperature_BTE", "Flux_BTE"])

    return kappa_eff_BTE, temp_BTE, flux_BTE, mesh


def convert_pores(pores):
    
    indices = np.argwhere(pores)

    indices = indices[:, [1, 0]]  

    # Calculate the centers for each pore
    pores_centers = (indices * (-20) +40).tolist()

    for p in pores_centers:
        p[0] = -p[0]
    
    return pores_centers

def convert_pores_new(pores):
    """
    Convert matrix of integers into pore positions with areas.
    Each non-zero entry becomes a pore, and its area is the square of the value.
    """
    indices = np.argwhere(pores)
    indices = indices[:, [1, 0]]  # switch x and y axis
    
    pores_with_area = []
    for idx in indices:
        x, y = idx
        value = pores[y, x]
        if value > 0:
            area = float(value) ** 2
            center_x, center_y = (x * -20) + 40, (y * -20) + 40
            pores_with_area.append((area, -center_x, center_y))  # flip x to match your convention
    
    return pores_with_area



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.tri as tri

    # --- run solver (ensure highfidelity_solver returns mesh along with kappa,temp,flux) ---
    pores = np.array(
        [1, 1, 0, 1, 1,
         1, 0, 0, 1, 1,
         0, 0, 0, 1, 0,
         0, 1, 1, 0, 0,
         1, 0, 0, 1, 1]
    ).reshape((1, 5, 5))

    # make sure your solver returns mesh as 4th item:
    # return kappa_eff_BTE, temp_BTE, flux_BTE, mesh
    kappa, temp, flux, mesh = highfidelity_solver(pores, step_size=5.0, save_show_res=True)

    print("Kappa:", kappa)
    print("temp shape:", np.asarray(temp).shape)
    print("flux shape:", np.asarray(flux).shape)
    print("mesh.nodes shape:", np.asarray(mesh.nodes).shape)
    print("mesh.elems shape:", np.asarray(mesh.elems).shape)

    # --- plotting function (same as before) ---
    def plot_temperature_unstructured(Temperatures, mesh, *,
                                  use_column=0,
                                  avg_columns=False,
                                  n_contour_levels=50,
                                  n_isoline_levels=20,
                                  mask_pores=True,
                                  cmap="viridis",
                                  fmt_isoline="%.1f",
                                  savepath=None):

        z = np.asarray(Temperatures)

        points = np.asarray(mesh.nodes)     # (n_nodes, 2)
        triangles = np.asarray(mesh.elems)  # (n_triangles, 3)
        x, y = points[:, 0], points[:, 1]

        # 🔥 Fix: if z is per-element, average values to nodes
        if z.shape[0] == triangles.shape[0] and z.shape[0] != points.shape[0]:
            print("Detected element-centered field; converting to node-centered.")
            z_node = np.zeros(points.shape[0])
            counts = np.zeros(points.shape[0])
            for tri, val in zip(triangles, z):
                for node in tri:
                    z_node[node] += val
                    counts[node] += 1
            z = z_node / counts
        elif z.shape[0] != points.shape[0]:
            raise ValueError(f"Length mismatch after check: field len {z.shape[0]} vs mesh nodes {points.shape[0]}")

        triang = tri.Triangulation(x, y, triangles)

        # --- Plot ---
        plt.figure(figsize=(6, 5))
        tpc = plt.tricontourf(triang, z, levels=n_contour_levels, cmap=cmap)
        cb = plt.colorbar(tpc, label="Temperature")
        cs = plt.tricontour(triang, z, levels=n_isoline_levels, colors="white", linewidths=0.7)
        plt.clabel(cs, inline=True, fmt=fmt_isoline, fontsize=8)

        plt.title("Temperature (BTE) on unstructured mesh")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal")
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300)
            print(f"Saved temperature plot → {savepath}")
        else:
            plt.show()


    # --- call plotting ---
    # Use the temperature field returned by solver (not flux). If solver temp is in solver.variables dict,
    # ensure you pass the correct array. Here we assume `temp` is the scalar field.
    plot_temperature_unstructured(temp, mesh, savepath="figures/paper/temp_bte.png")