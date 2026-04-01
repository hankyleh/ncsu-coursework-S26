from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem


degree = 3
h = 2
gamma = -1.0

l2results = np.array([])
maxabs    =  np.array([])

for h in [4, 8, 16, 32]:
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(h, h),
        cell_type=mesh.CellType.triangle,
    )
    n = ufl.FacetNormal(msh)
    V = fem.functionspace(msh, ("Lagrange", degree))
    # -



    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(
        msh,
        dim=fdim,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)



    # +
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 2*(ufl.pi**2)*ufl.cos(x[0]*ufl.pi)*ufl.cos(x[1]*ufl.pi)
    # g = ufl.sin(5 * x[0]) + gamma
    u_exact = ufl.cos(x[0]*ufl.pi)*ufl.cos(x[1]*ufl.pi)
    g = gamma*(u_exact) + ufl.dot(ufl.grad(u_exact), n)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx) + gamma*(ufl.inner(u, v) * ufl.ds)
    L = (ufl.inner(f, v) * ufl.dx) + (ufl.inner(g, v) * ufl.ds)
    # -


    # +
    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix="demo_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
    )
    uh = problem.solve()
    assert isinstance(uh, fem.Function)
    # -

    # The solution can be written to a {py:class}`XDMFFile
    # <dolfinx.io.XDMFFile>` file visualization with [ParaView](https://www.paraview.org/)
    # or [VisIt](https://visit-dav.github.io/visit-website/):

    # +
    out_folder = Path("out_poisson")
    out_folder.mkdir(parents=True, exist_ok=True)
    with io.VTXWriter(msh.comm, out_folder / "poisson.bp", [uh], engine="BP4") as vtx:
        vtx.write(0.0)
    # -

    # and displayed using [pyvista](https://docs.pyvista.org/).


    exact_solution = lambda x:  ufl.cos(x[0]*ufl.pi) * ufl.cos(x[1]*ufl.pi)
    exact_solution_np = lambda x: np.cos(x[0] * np.pi) * np.cos(x[1] * np.pi)

    # 2. Create a fine grid for high-resolution evaluation
    # Reduced from 500 to 100, because 100x100 = 10,000 evaluation points.
    # 500x500 would be 250,000 points and might take a moment to compute collisions!
    num_eval_points = 100 
    x_1d = np.linspace(0, 1, num_eval_points)
    y_1d = np.linspace(0, 1, num_eval_points)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)

    # FEniCSx always expects a 3D array of points: shape (3, N)
    points = np.zeros((3, x_grid.size))
    points[0, :] = x_grid.flatten()
    points[1, :] = y_grid.flatten() # Now populating the y-coordinates

    # 3. Compute collisions to map points to mesh cells
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

    # Extract both x and y coordinates for evaluating the exact solution
    x_fine_plot = valid_points[:, 0]
    y_fine_plot = valid_points[:, 1]

    # 4. Evaluate the FEM solution (this captures the full polynomial degree)
    u_fine = uh.eval(valid_points, cells).flatten()

    # Evaluate the exact solution (passing both x and y arrays)
    u_exact_fine = exact_solution_np((x_fine_plot, y_fine_plot))

    # 5. Calculate Errors
    # High-resolution pointwise error for plotting
    pointwise_error = np.abs(u_fine - u_exact_fine)

    # Formal L2 Error (Standard for FEM analysis)
    # ufl.SpatialCoordinate automatically handles the 2D spatial coordinates here
    u_ex_sym = exact_solution(ufl.SpatialCoordinate(msh)) 
    error_form = fem.form(ufl.inner(uh - u_ex_sym, uh - u_ex_sym) * ufl.dx)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))


    l2results = np.append(l2results, error_L2)
    maxabs   = np.append(maxabs, np.max(pointwise_error))




    # +
    try:
        import pyvista

        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = uh.x.array.real
        grid.set_active_scalars("u")
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_scalar()
        plotter.add_mesh(warped)
        if pyvista.OFF_SCREEN:
            plotter.screenshot(out_folder / f"prob3_n{h}_d{degree}_g{gamma}.png")
        else:
            plotter.show()
            plotter.screenshot(out_folder / f"prob3_n{h}_d{degree}_g{gamma}.png")
    except ModuleNotFoundError:
        print("'pyvista' is required to visualise the solution.")
        print("To install pyvista with pip: 'python3 -m pip install pyvista'.")
print(f"Degree {degree}:")
print(f"elements: 4 8 16 32 square * 2")
print("l2 error")
for i in l2results:
    print(f"{i:.3e}")
print("max absolute ptwise")
for i in maxabs:
    print(f"{i:.3e}")