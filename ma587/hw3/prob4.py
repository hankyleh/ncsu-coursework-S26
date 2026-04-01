from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem
import basix.ufl

degree = 3 # Degree 2 is standard for the Ciarlet-Raviart mixed formulation
nx = 12    # Increased from 2 to properly resolve the geometry

l2results = np.array([])
maxabs    =  np.array([])

for nx in [4, 8, 16, 32]:
    # ---------------------------------------------------------
    # 1. Mesh and Mixed Function Space
    # ---------------------------------------------------------
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(nx, nx),
        cell_type=mesh.CellType.triangle,
    )

    # Define a Mixed Element: [Lagrange for u, Lagrange for w]
    el = basix.ufl.element("Lagrange", msh.basix_cell(), degree)
    mel = basix.ufl.mixed_element([el, el])
    V = fem.functionspace(msh, mel)

    # ---------------------------------------------------------
    # 2. Boundary Conditions
    # ---------------------------------------------------------
    # We only apply u = 0 on the boundary. du/dn = 0 is naturally satisfied.
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        msh,
        dim=fdim,
        marker=lambda x: np.full(x.shape[1], True) # Selects the entire boundary
    )

    # Locate DOFs directly in the 'u' subspace (index 0) without collapsing.
    # This returns a 1D array of DOFs, perfectly matching the ScalarType input.
    dofs = fem.locate_dofs_topological(V.sub(0), fdim, facets)
    bc = fem.dirichletbc(ScalarType(0), dofs, V.sub(0))

    # ---------------------------------------------------------
    # 3. Variational Problem
    # ---------------------------------------------------------
    (u, w) = ufl.TrialFunctions(V)
    (v, tau) = ufl.TestFunctions(V)
    x = ufl.SpatialCoordinate(msh)

    # Define exact solution symbolically
    exact_solution_ufl = lambda x: (ufl.sin(ufl.pi * x[0])**2) * (ufl.sin(ufl.pi * x[1])**2)
    u_ex = exact_solution_ufl(x)

    f = ufl.div(ufl.grad(ufl.div(ufl.grad(u_ex))))

    a = (ufl.inner(ufl.grad(w), ufl.grad(v)) * ufl.dx) \
    + (ufl.inner(ufl.grad(u), ufl.grad(tau)) * ufl.dx) \
    - (ufl.inner(w, tau) * ufl.dx)

    L = ufl.inner(f, v) * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="demo_biharmonic_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps", # <-- Tell PETSc to use MUMPS
            "ksp_error_if_not_converged": True
        },
    )
    uh = problem.solve()
    assert isinstance(uh, fem.Function)

    # Split the mixed function and collapse 'u' into its own standard Function
    uh_u, uh_w = uh.split()
    uh_u = uh_u.collapse()
    uh_u.name = "u"

    out_folder = Path("p4")
    out_folder.mkdir(parents=True, exist_ok=True)
    with io.VTXWriter(msh.comm, out_folder / "biharmonic.bp", [uh_u], engine="BP4") as vtx:
        vtx.write(0.0)

    # ---------------------------------------------------------
    # 5. High-Resolution Evaluation & Error Checking
    # ---------------------------------------------------------
    exact_solution_np = lambda x: (np.sin(np.pi * x[0])**2) * (np.sin(np.pi * x[1])**2)

    num_eval_points = 100 
    x_1d = np.linspace(0, 1, num_eval_points)
    y_1d = np.linspace(0, 1, num_eval_points)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)

    points = np.zeros((3, x_grid.size))
    points[0, :] = x_grid.flatten()
    points[1, :] = y_grid.flatten()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)

    cells = []
    valid_points = []
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            valid_points.append(point)
            cells.append(colliding_cells.links(i)[0])

    valid_points = np.array(valid_points)
    x_fine_plot = valid_points[:, 0]
    y_fine_plot = valid_points[:, 1]

    # Evaluate the collapsed 'u' solution
    u_fine = uh_u.eval(valid_points, cells).flatten()
    u_exact_fine = exact_solution_np((x_fine_plot, y_fine_plot))

    pointwise_error = np.abs(u_fine - u_exact_fine)

    error_form = fem.form(ufl.inner(uh_u - u_ex, uh_u - u_ex) * ufl.dx)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))

    l2results = np.append(l2results, error_L2)
    maxabs   = np.append(maxabs, np.max(pointwise_error))

    # ---------------------------------------------------------
    # 6. PyVista Plotting
    # ---------------------------------------------------------
    try:
        import pyvista

        # Create mesh geometry based on the collapsed u space
        vtk_cells, vtk_types, vtk_x = plot.vtk_mesh(uh_u.function_space)
        grid = pyvista.UnstructuredGrid(vtk_cells, vtk_types, vtk_x)
        grid.point_data["u"] = uh_u.x.array.real
        grid.set_active_scalars("u")
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        
        if pyvista.OFF_SCREEN:
            plotter.screenshot(out_folder / f"prob4_n{nx}_d{degree}.png")
        else:
            plotter.show()
            plotter.screenshot(out_folder / f"prob4_n{nx}_d{degree}.png")
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution.")




print(f"Degree {degree}:")
print(f"elements: 4 8 16 32 square * 2")
print("l2 error")
for i in l2results:
    print(f"{i:.3e}")
print("max absolute ptwise")
for i in maxabs:
    print(f"{i:.3e}")