import matplotlib.pyplot as plt

from fenics import *
from tools.elasticity import *
from tools.damage import *
import sympy as sp

#----------------------------------------------------------------------#
# Set some Fenics parameters
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["representation"] = 'uflacs'
parameters["linear_algebra_backend"]          = "PETSc"
set_log_level(LogLevel.CRITICAL)
info(parameters, False)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Compute the symbolic expression for eigenvalues by sympy
T = sp.Matrix(2, 2, lambda i, j: sp.Symbol('T[%d, %d]' % (i, j), real=True))
eig_expr = T.eigenvects()   # ((v, multiplicity, [w])...)

eigv = [e[0] for e in eig_expr]
eigw = [e[-1][0] for e in eig_expr]

eigv_expr = map(str, eigv)
eigw_expr = [[str(e[0]), str(e[1])] for e in eigw]

# UFL operator for eigenvalues of 2x2 matrix, a pair of scalars
def eigv(T): return map(eval, eigv_expr)

# UFL operator for eigenvectors of 2x2 matrix, a pair of vectors
def eigw(T): return [as_vector(map(eval, vec)) for vec in eigw_expr]
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
topBC_y   = DirichletBC(W.sub(1), Constant(0.0), top)
topBC_x   = DirichletBC(W.sub(0), ud, top)

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

histold = energy_density(uold, lmbda, mu)
histnew = energy_density(unew, lmbda, mu)
hist    = Max(histold, histnew)

Id = ((2.0*hist + gc/lc)*dot(d,s) + gc*lc*inner(nabla_grad(d), nabla_grad(s)) - 2.0*hist*s)*dx
Iu = ((1.0-dnew)*(1.0-dnew)+1E-6)*inner(sigma(u, lmbda, mu), epsilon(v))*dx

Ad, Ld = lhs(Id), rhs(Id)
Au, Lu = lhs(Iu), rhs(Iu)

u, d = Function(W), Function(V)

prob_dmge = LinearVariationalProblem(Ad, Ld, d)
prob_disp = LinearVariationalProblem(Au, Lu, u, bcs)

sigma = sigma_spectral_split(u, uold, dnew, lmbda, mu)

"""
solver_dmge = LinearVariationalSolver(prob_dmge)
solver_disp = LinearVariationalSolver(prob_disp)

solver_dmge.parameters["linear_solver"] = "gmres"
solver_dmge.parameters["preconditioner"] = "hypre_euclid"
solver_dmge.parameters["krylov_solver"]["relative_tolerance"] = 1E-6
solver_dmge.parameters["krylov_solver"]["absolute_tolerance"] = 1E-6

solver_disp.parameters["linear_solver"] = "gmres"
solver_disp.parameters["preconditioner"] = "hypre_euclid"
solver_disp.parameters["krylov_solver"]["relative_tolerance"] = 1E-6
solver_disp.parameters["krylov_solver"]["absolute_tolerance"] = 1E-6
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Staggered algorithm
nsteps = 1000
delta = 1E-4

for n in range(nsteps):
    ud.t += delta

    dnew.assign(d)
    uold.assign(u)

    solver_disp.solve()

    unew.assign(u)

    solver_dmge.solve()

vtkfile_d = File('damagefield.pvd')
vtkfile_u = File('displacement.pvd')

vtkfile_d << d
vtkfile_u << u

plt.figure()
plot(d, cmap='jet')
plt.show()
"""
