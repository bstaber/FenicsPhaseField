import matplotlib.pyplot as plt

from fenics import *
from elasticity import *
from damage import *
from fenics import *

#parameters
parameters["linear_algebra_backend"] = "PETSc"
set_log_level(LogLevel.PROGRESS)
info(parameters, False)

#load mesh and define functional spaces
mesh = Mesh('meshes/mesh_fenics.xml')
mesh = refine(mesh)
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

#variational problems
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

#staggered algorithm
nsteps = 500
delta = 1E-4

for n in range(nsteps):
    ud.t += delta

    dnew.assign(d)
    uold.assign(u)

    solver_disp.solve()

    unew.assign(u)

    solver_dmge.solve()

plt.figure()
plot(d, cmap='jet')
plt.show()

ALE.move(mesh, u)
plt.figure()
plot(mesh)
plt.show()
