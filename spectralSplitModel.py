import matplotlib.pyplot as plt

from fenics import *
from tools.elasticity import *

#----------------------------------------------------------------------#
# Set some Fenics parameters
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["representation"] = 'uflacs'
parameters["linear_algebra_backend"]          = "PETSc"
set_log_level(LogLevel.PROGRESS)
info(parameters, False)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Load mesh and define functional spaces
mesh = Mesh('meshes/mesh_fenics.xml')
mesh = refine(mesh)
V = FunctionSpace(mesh, 'Lagrange', 1)
W = VectorFunctionSpace(mesh, 'Lagrange', 1, 2)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define boundary conditions
ud = Expression("t", t=0.0, degree=1)

def bottom(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[1]) < tol

def top(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[1]-1.0) < tol

def right(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[0]-1.0) < tol

def left(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[0]) < tol


bottomBCs = DirichletBC(W, Constant((0.0, 0.0)), bottom)
topBC_y   = DirichletBC(W.sub(1), ud, top)
topBC_x   = DirichletBC(W.sub(0), Constant(0.0), top)
bcs = [bottomBCs, topBC_x, topBC_y]
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Variational problems
lmbda, mu = 121.15e3, 80.77e3
gc, lc    = 2.7, 0.0075

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

d, s = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)

uold, unew = Function(W), Function(W)
dnew       = Function(V)

histold = energy_density_positive(uold, lmbda, mu)
histnew = energy_density_positive(unew, lmbda, mu)
hist    = Max(histold, histnew)

Id = ((2.0*hist + gc/lc)*dot(d,s) + gc*lc*inner(nabla_grad(d), nabla_grad(s)) - 2.0*hist*s)*dx
Iu = inner(sigma_spectral_split(u, uold, dnew, lmbda, mu), epsilon_voigt(v))*dx

Ad, Ld = lhs(Id), rhs(Id)
Au, Lu = lhs(Iu), rhs(Iu)

u, d = Function(W), Function(V)

prob_dmge = LinearVariationalProblem(Ad, Ld, d)
prob_disp = LinearVariationalProblem(Au, Lu, u, bcs)

solver_dmge = LinearVariationalSolver(prob_dmge)
solver_disp = LinearVariationalSolver(prob_disp)

"""
solver_dmge.parameters["linear_solver"] = "gmres"
solver_dmge.parameters["preconditioner"] = "hypre_euclid"
solver_dmge.parameters["krylov_solver"]["relative_tolerance"] = 1E-6
solver_dmge.parameters["krylov_solver"]["absolute_tolerance"] = 1E-6

solver_disp.parameters["linear_solver"] = "gmres"
solver_disp.parameters["preconditioner"] = "hypre_euclid"
solver_disp.parameters["krylov_solver"]["relative_tolerance"] = 1E-6
solver_disp.parameters["krylov_solver"]["absolute_tolerance"] = 1E-6
"""
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Staggered algorithm
nsteps = 250
delta  = 1E-4

for n in range(nsteps):
    ud.t += delta

    dnew.assign(d)
    uold.assign(u)

    solver_disp.solve()

    unew.assign(u)

    solver_dmge.solve()

plot(d, cmap='jet', range_min=0., range_max=1.)
plt.show()

vtkfile_d = File('damagefield.pvd')
vtkfile_u = File('displacement.pvd')

vtkfile_d << d
vtkfile_u << u
