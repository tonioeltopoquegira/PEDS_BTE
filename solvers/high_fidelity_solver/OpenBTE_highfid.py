# High fidelity solver using the open-source package OpenBTE.
# OpenBTE, an efficient solver for the steady-state phonon BTE in multidimensional structures.
# This tool is interfaced to first-principles calculations, thus it unlocks the calculations of thermal-related properties with no fitting-parameters

import numpy as np
import os

# OpenBTE
from openbte import load_rta, RTA2DSym, Geometry, BTE_RTA, Fourier, rectangle, get_mesh
from openbte.objects import BoundaryConditions, OpenBTEResults, EffectiveThermalConductivity

# --- optional SciPy-based interpolation helpers ---
try:
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_INTERP = True
except Exception:
    SCIPY_INTERP = False

try:
    from scipy.spatial import cKDTree
    SCIPY_KDTREE = True
except Exception:
    SCIPY_KDTREE = False



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



    return kappa_eff_BTE, temp_BTE, flux_BTE, results


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

# -------------------------
# Utilities for BTE node extraction (same approach as your plot function)
# -------------------------
def get_bte_nodes_elems_nodevars(results, varname='Temperature_BTE', repeat=(1,1,1)):
    """
    Extract duplicated nodes, elements and nodal variable values for a variable
    (used to interpolate/plot).
    Returns: nodes (N x 2), elems (M x 3), node_var (N,)
    """
    tmp_variables = results.get_variables()
    if varname not in tmp_variables:
        raise KeyError(f"{varname} not found in results variables: {list(tmp_variables.keys())}")
    variables = {varname: tmp_variables[varname]}

    nodes, elems = results.mesh.duplicate_cells(variables, repeat)
    node_variables = results.mesh.cell_to_node(variables, nodes, elems)
    node_var = node_variables[varname]['data'].astype(float)
    return nodes, np.array(elems, dtype=int), node_var

# -------------------------
# Interpolation helper: interpolate Fourier grid (regular) onto target points
# -------------------------
def interp_fourier_to_points(fourier_T, domain_L, target_xy):
    """
    Attempt to interpolate a regular Fourier grid to target (N,2) xy points.
    - fourier_T: (ny, nx) 2D array (no batch dimension)
    - domain_L: physical domain length (assumed 0..L in both x and y)
    - target_xy: (N,2) array of [x,y] query points
    Returns array (N,) of interpolated temperatures.
    NOTE: If scipy RegularGridInterpolator not available, falls back to nearest-neighbor using KDTree (if available),
    or simply broadcasts the mean (last resort).
    """
    ny, nx = fourier_T.shape
    # assume grid points are cell centers evenly spaced between 0..L
    grid_x = np.linspace(0, domain_L, nx)
    grid_y = np.linspace(0, domain_L, ny)

    pts = np.vstack(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T  # (nx*ny, 2)
    vals = fourier_T.ravel()

    if SCIPY_INTERP:
        try:
            interp = RegularGridInterpolator((grid_y, grid_x), fourier_T, bounds_error=False, fill_value=np.nan)
            pts_query = np.asarray(target_xy)
            q = interp(pts_query[:, [1,0]] if False else pts_query)  # assume target is (x,y)
            # Note: RegularGridInterpolator expects axes in same order; we used (y,x) above -> tried to be careful
            return q
        except Exception:
            pass

    # fallback: nearest neighbor with KDTree
    if SCIPY_KDTREE:
        try:
            tree = cKDTree(pts)
            d, ii = tree.query(target_xy)
            return vals[ii]
        except Exception:
            pass

    # last resort: return mean (and warn)
    print("Warning: interpolation fallback used; install scipy for better interpolation.")
    return np.full(len(target_xy), np.nanmean(vals))

# -------------------------
# Main script: run both solvers and plot + compute error
# -------------------------
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.tri as tri
    import sys
    from pathlib import Path

    # Adjust this number if your file location differs.
    # For file: <project_root>/solvers/high_fidelity_solver/OpenBTE_highfid.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]   # two levels up -> project root
    sys.path.insert(0, str(PROJECT_ROOT))

    # now safe to import
    from solvers.low_fidelity_solvers.base_conductivity_grid_converter import conductivity_original_wrapper
    from solvers.low_fidelity_solvers.fourier import fourier_solver, plot_temperature_fourier


    # example pore pattern (your provided 5x5)
    pores = np.array([
        1,1,0,0,1,
        1,0,0,1,1,
        1,1,0,1,0,
        0,1,0,0,1,
        1,0,0,1,1
    ]).reshape((5,5))

    # If you want the specific pattern used in your snippet replace above with:
    # pores = np.array([0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1, 0,0,0,1,0]).reshape((5,5))

    # ---------- Low-fidelity: conductivity grid -> Fourier solver ----------
    L = 100  # domain size used in both parts (assumption)
    # conductivity_original_wrapper seems to expect a batch dim; pass pores[None,...]
    grid_custom = conductivity_original_wrapper(pores[None, :, :], L)  # shape (batch, ny, nx) expected
    # run Fourier solver
    T_custom, kappa_fourier = fourier_solver(grid_custom)
    # T_custom shape likely (batch, ny, nx)
    print("Fourier effective kappa:", kappa_fourier)

    # plot Fourier temperature (uses your plotting function)
    try:
        plot_temperature_fourier(T_custom, grid_custom, index=0)
    except Exception as e:
        print("plot_temperature_fourier failed:", e)

    # ---------- High-fidelity: BTE ----------
    print("Running high-fidelity BTE solver (this can take some time)...")
    kappa_bte, temp_bte, flux_bte, results = highfidelity_solver(pores, step_size=1.5, save_show_res=True)
    print("BTE effective kappa:", kappa_bte)

    # plot BTE temperature using your function
    try:
        # reuse your plot_bte_temperature (you included it earlier) — but we reconstruct here as a simple call:
        # The results object holds the mesh and variables; the implementation below mirrors your plotting function.
        plot_bte = True
        if plot_bte:
            # Extract nodes & triangles & node temp (duplicating the steps used in your plot function)
            nodes, elems, node_temp = get_bte_nodes_elems_nodevars(results, varname='Temperature_BTE', repeat=(1,1,1))
            x, y = nodes[:,0], nodes[:,1]
            triang = tri.Triangulation(x, y, np.array(elems))
            fig, ax = plt.subplots(figsize=(6,5))
            tcf = ax.tricontourf(triang, node_temp, levels=50, cmap='viridis_r')
            plt.colorbar(tcf, ax=ax, label="Temperature_BTE")
            ax.tricontour(triang, node_temp, levels=np.linspace(np.nanmin(node_temp), np.nanmax(node_temp), 25),
                           colors='white', linewidths=0.7)
            ax.set_aspect('equal')
            ax.set_title("Temperature_BTE")
            ax.set_xlabel("x direction")
            ax.set_ylabel("y direction")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("plot_bte_temperature failed:", e)

    # ---------- Interpolate Fourier field onto BTE nodes and compute error ----------
    # Prepare Fourier 2D array (no batch)
    fourier_T2d = np.squeeze(T_custom[0]) if T_custom.ndim == 3 else T_custom
    # Extract BTE nodes & node temperatures (already done above but re-call for clarity)
    nodes, elems, node_temp = get_bte_nodes_elems_nodevars(results, varname='Temperature_BTE', repeat=(1,1,1))
    # nodes assumed shape (N,2) with columns [x,y]
    # Interpolate Fourier onto nodes:
    interp_vals = interp_fourier_to_points(fourier_T2d, domain_L=L, target_xy=nodes)

    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt

    # Compute error field
    valid_mask = ~np.isnan(interp_vals)
    if not np.any(valid_mask):
        print("Warning: no valid interpolated points. Can't compute error field.")
    else:
        diff = node_temp - interp_vals
        # error statistics
        l2 = np.linalg.norm(diff[valid_mask])
        l2_rel = l2 / np.linalg.norm(node_temp[valid_mask]) if np.linalg.norm(node_temp[valid_mask]) > 0 else np.nan
        max_abs = np.nanmax(np.abs(diff))
        mean_abs = np.nanmean(np.abs(diff))
        print(f"Temperature error stats (BTE nodes - Fourier interp): "
            f"L2={l2:.6g}, relative L2={l2_rel:.6g}, "
            f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}")

        # define normalization so white = 0
        norm = TwoSlopeNorm(vmin=np.nanmin(diff), vcenter=0.0, vmax=np.nanmax(diff))

        # plot
        fig, ax = plt.subplots(figsize=(6,5))
        tcf = ax.tricontourf(triang, diff, levels=50, cmap='RdBu_r', norm=norm)
        cbar = plt.colorbar(tcf, ax=ax, label='T_BTE - T_Fourier_interp')
        ax.set_aspect('equal')
        ax.set_title('Temperature Error (BTE - Fourier_interp)')
        ax.set_xlabel("x direction")
        ax.set_ylabel("y direction")
        plt.tight_layout()
        plt.show()

    # ---------- Final kappas and simple relative error between kappas ----------
    print("---- Summary ----")
    print(f"Fourier kappa: {kappa_fourier}")
    print(f"BTE kappa:     {kappa_bte}")
    try:
        rel_kappa_err = (kappa_bte - kappa_fourier) / float(kappa_bte)
    except Exception:
        rel_kappa_err = np.nan
    print(f"Relative kappa difference (BTE - Fourier) / BTE = {rel_kappa_err:.6g}")

    
    # Continue
