from fenics import *
from elasticity import *

import matplotlib.pyplot as plt

#parameters
parameters["linear_algebra_backend"] = "PETSc"
prm = parameters["krylov_solver"]
prm["absolute_tolerance"]  = 1E-6
prm["relative_tolerance"]  = 1E-6
prm["maximum_iterations"]  = 100
prm["monitor_convergence"] = True
set_log_level(LogLevel.PROGRESS)

info(parameters, True)

#load mesh
mesh = Mesh('meshes/mesh_fenics.xml')
V = FunctionSpace(mesh, 'Lagrange', 1)
W = VectorFunctionSpace(mesh, 'Lagrange', 1, 2)

#define boundary conditions

ud = Expression("t", t=0.1, degree=1)

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

# damage-gradient variational problem

'''
TO DO
'''

# elasticity variational problem

'''
TO DO : spectral split, degradation
'''

lmbda, mu = 121.15e3, 80.77e3

v = TestFunction(W)
u = TrialFunction(W)

au = inner(sigma(u,lmbda,mu), epsilon(v))*dx
lu = dot(Constant((0.0,0.0)),v)*dx

A, b = assemble_system(au, lu, bcs)
solver = PETScKrylovSolver('cg', 'hypre_euclid')
solver.parameters["absolute_tolerance"]  = 1E-8
solver.parameters["relative_tolerance"]  = 1E-8
solver.parameters["maximum_iterations"]  = 100
solver.parameters["monitor_convergence"] = True
solver.set_operator(A)

"""
u = Function(W)
solver.solve(u.vector(), b)
"""
