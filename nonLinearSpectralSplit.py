import matplotlib.pyplot as plt

from fenics import *
from tools.elasticity import *

#----------------------------------------------------------------------#
# Set some Fenics parameters
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["cpp_optimize"]   = True
"""
parameters["form_compiler"]["representation"] = "uflacs"
"""
parameters["linear_algebra_backend"]          = "PETSc"
set_log_level(LogLevel.PROGRESS)
info(parameters, False)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Load mesh and define functional spaces
mesh = Mesh("meshes/mesh_fenics.xml")
mesh = refine(mesh)
dim  = mesh.topology().dim()
V    = FunctionSpace(mesh, 'Lagrange', 1)
W    = VectorFunctionSpace(mesh, 'Lagrange', 1, dim)
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
bcs       = [bottomBCs, topBC_x, topBC_y]
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define weak forms
lmbda, mu = 121.15e3, 80.77e3
gc, lc    = 2.7, 0.0075

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

d,  s = TrialFunction(V), TestFunction(V)
du, v = TrialFunction(W), TestFunction(W)

uold, unew, u_ = Function(W), Function(W), Function(W)
dnew           = Function(V)

histold = energy_density_positive(uold, lmbda, mu)
histnew = energy_density_positive(unew, lmbda, mu)
hist    = Max(histold, histnew)

Id = ((2.0*hist + gc/lc)*dot(d,s) + gc*lc*inner(nabla_grad(d), nabla_grad(s)) - 2.0*hist*s)*dx
Fu = inner(sigma_spectral_split(u_, dnew, lmbda, mu), epsilon(v))*dx
Ju = derivative(Fu, u_, du)

Ad, Ld = lhs(Id), rhs(Id)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define variational problems
d = Function(V)
prob_dmge = LinearVariationalProblem(Ad, Ld, d)
prob_disp = NonlinearVariationalProblem(Fu, u_, bcs, Ju)

solver_dmge = LinearVariationalSolver(prob_dmge)
solver_disp = NonlinearVariationalSolver(prob_disp)

prm = solver_disp.parameters
prm["newton_solver"]["absolute_tolerance"]   = 1E-8
prm["newton_solver"]["relative_tolerance"]   = 1E-7
prm["newton_solver"]["maximum_iterations"]   = 5
prm["newton_solver"]["relaxation_parameter"] = 1.0
"""
prm["linear_solver"]  = "gmres"
prm["preconditioner"] = "ilu"
prm["krylov_solver"]["absolute_tolerance"] = 1E-9
prm["krylov_solver"]["relative_tolerance"] = 1E-7
prm["krylov_solver"]["maximum_iterations"] = 1000
prm["krylov_solver"]["gmres"]["restart"]   = 40
prm["krylov_solver"]["preconditioner"]["ilu"]["fill_level"] = 0
"""
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Staggered algorithm
nsteps = 250
delta  = 1E-5

d.vector().zero()
dnew.assign(d)

ud.t += delta

uold.vector().zero()
solver_disp.solve()
unew.assign(u_)


for n in range(nsteps):
    ud.t += delta

    solver_dmge.solve()
    dnew.assign(d)

    uold.assign(u_)
    solver_disp.solve()
    unew.assign(u_)

    plot(d, cmap='jet')
    plt.draw()
    plt.pause(0.0001)
#----------------------------------------------------------------------#
