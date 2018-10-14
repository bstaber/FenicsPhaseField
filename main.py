from fenics import *
import matplotlib.pyplot as plt

#parameters
parameters["linear_algebra_backend"] = "PETSc"
prm = parameters["krylov_solver"]
prm["absolute_tolerance"]  = 1E-10
prm["relative_tolerance"]  = 1E-6
prm["maximum_iterations"]  = 1000
prm["monitor_convergence"] = False
set_log_level(LogLevel.PROGRESS)

"""info(parameters, True)"""

#load mesh
mesh = Mesh('meshes/mesh_fenics.xml')
V = FunctionSpace(mesh, 'Lagrange', 1)
W = VectorFunctionSpace(mesh, 'Lagrange', 1, 2)

#define boundary conditions

ud = Expression("t", t=0.0, degree=1)

def bottom(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[1]) < tol

def top(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[1]-1.0) < tol

bottomBCs = DirichletBC(W, Constant((0.0, 0.0)), bottom)
topBC_x   = DirichletBC(W.sub(0), ud, top)
topBC_y   = DirichletBC(W.sub(1), Constant(0.0), top)

bcs = [bottomBCs, topBC_x, topBC_y]

#define variational problems

d, u = TrialFunction(V), TrialFunction(W)
s, v = TestFunction(V), TestFunction(W)

#solver
