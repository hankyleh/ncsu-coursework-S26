from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, geometry
from dolfinx.fem.petsc import LinearProblem
import matplotlib
import matplotlib.pyplot as plt


degree = 2
num_elements = 8


l2results = np.array([])
maxabs    =  np.array([])

for num_elements in [4, 7, 16, 32]:
    msh = mesh.create_unit_interval(
        comm=MPI.COMM_WORLD,
        nx=num_elements,
    )
    V = fem.functionspace(msh, ("Lagrange", degree))

    tdim = msh.topology.dim
    fdim = tdim - 1
    facets = mesh.locate_entities_boundary(
        msh,
        dim=fdim,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
    )


    dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

    # +
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    # f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    f = 1.0
    # g = ufl.sin(5 * x[0])
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds
    L = ufl.inner(f, v) * ufl.dx
    # -

    # +
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="demo_poisson_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
    )
    uh = problem.solve()
    assert isinstance(uh, fem.Function)
    # -

    # +
    out_folder = Path("out_poisson")
    out_folder.mkdir(parents=True, exist_ok=True)
    with io.VTXWriter(msh.comm, out_folder / "poisson.bp", [uh], engine="BP4") as vtx:
        vtx.write(0.0)
    # -


    exact_solution = lambda x: 0.5 * x[0] * (1 - x[0])

    num_eval_points = 500
    x_fine = np.linspace(0, 1, num_eval_points)

    points = np.zeros((3, num_eval_points))
    points[0, :] = x_fine


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


    u_fine = uh.eval(valid_points, cells).flatten()

    u_exact_fine = exact_solution((x_fine_plot,))



    pointwise_error = np.abs(u_fine - u_exact_fine)

    u_ex_sym = exact_solution(ufl.SpatialCoordinate(msh)) 
    error_form = fem.form(ufl.inner(uh - u_ex_sym, uh - u_ex_sym) * ufl.dx)
    error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM))
 

    l2results = np.append(l2results, error_L2)
    maxabs   = np.append(maxabs, np.max(pointwise_error))
    plt.figure()
    plt.plot(figsize=(6, 5))

    ax1 =plt.gca()

    # Plot 1: The Solutions
    ax1.plot(x_fine_plot, u_exact_fine, 'k--', linewidth=2, label="Exact Solution")
    ax1.plot(x_fine_plot, u_fine, 'b-', alpha=0.7, linewidth=2, label=f"FEM Solution (Degree {degree}, Nx {num_elements})")
    ax1.set_title("1D Poisson Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)



    plt.tight_layout()
    plt.savefig(out_folder / f"prob1_n{num_elements}_d{degree}.png", dpi=300)
    # plt.show()
print(f"Degree {degree}:")
print(f"elements: 4 8 16 32")
print("l2 error")
for i in l2results:
    print(f"{i:.3e}")
print("max absolute ptwise")
for i in maxabs:
    print(f"{i:.3e}")