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

def damageHistory(u):
    str_ele = 0.5*(grad(u) + grad(u).T)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC

def damageFunction(d):
    tol = 1E-6
    return (1.0-d)*(1.0-d) + tol

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return 2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(2)

gc, lc = 2.7, 0.0075
lmbda, mu = 1.0, 10.0

dold, d, s = TrialFunction(V), TrialFunction(V), TestFunction(V)
uold, u, v = TrialFunction(W), TrialFunction(W), TestFunction(W)

ad = (2.0*damageHistory(uold) + (gc/lc)*dot(d,s))*dx + gc*lc*inner(nabla_grad(d),nabla_grad(s))*dx
ld = 2.0*damageHistory(uold)*s*dx;

au = damageFunction(dold)*inner(nabla_grad(v),sigma(u))*dx

#solver
