import matplotlib.pyplot as plt

from fenics import *
from tools.elasticity import *

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
# Load mesh and define functional spaces
mesh = Mesh('meshes/srb.xml')
"""
mesh = refine(mesh)
"""
V = FunctionSpace(mesh, 'Lagrange', 1)
W = VectorFunctionSpace(mesh, 'Lagrange', 1, mesh.topology().dim())
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define boundary conditions and domains
ud = Expression("t", t=0.0, degree=1)

tol = 1E-10
class Steel(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.045, 0.055))

steel = Steel()

domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
steel.mark(domains, 1)

def right(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[0]-1.0) < tol and between(x[1], (0.045, 0.055))


def left(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[0]) < tol and between(x[1], (0.045, 0.055))

def bottom(x, on_boundary):
    tol = 1E-8
    return on_boundary and abs(x[1]) < tol


bcs = [DirichletBC(W, Constant((0.0, 0.0)), left),
       DirichletBC(W.sub(0), ud, right),
       DirichletBC(W.sub(1), Constant(0.0), right)]
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define material parameters
lmbda0, mu0  = 121.15e3, 80.77e3
lmbda1, mu1  = 121.15e4, 80.77e4
gc0, gc1, lc = 0.5, 50.0, 0.0005
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Define new measures associated with the interior domains and exterior
# boundaries

dx = Measure("dx", domain=mesh, subdomain_data=domains)

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

d, s = TrialFunction(V), TestFunction(V)
u, v = TrialFunction(W), TestFunction(W)

uold, unew = Function(W), Function(W)
dnew       = Function(V)

def energy_density_positive_bulk(u):
    e     = epsilon(u)
    tr_e  = tr(e)
    return 0.5*conditional(gt(tr_e,0.0),tr_e*tr_e,0.0)

def energy_density_positive_shear(u):
    e     = epsilon(u)
    tr_e  = tr(e)
    det_e = det(e)
    disc  = sqrt(tr_e*tr_e/4.0 - det_e)
    v1    = 0.5*tr_e + disc
    v2    = 0.5*tr_e - disc
    v1p   = conditional(gt(v1,0.0),v1,0.0)
    v2p   = conditional(gt(v2,0.0),v2,0.0)
    return v1p*v1p + v2p*v2p

energyold_bulk = energy_density_positive_bulk(uold)
energynew_bulk = energy_density_positive_bulk(unew)

energyold_shear = energy_density_positive_shear(uold)
energynew_shear = energy_density_positive_shear(unew)

histold_0 = lmbda0*energyold_bulk + mu0*energyold_shear
histnew_0 = lmbda0*energynew_bulk + mu0*energynew_shear

histold_1 = lmbda1*energyold_bulk + mu1*energyold_shear
histnew_1 = lmbda1*energynew_bulk + mu1*energynew_shear

hist_0 = Max(histold_0, histnew_0)
hist_1 = Max(histold_1, histnew_1)

Id = ((2.0*hist_0 + gc0/lc)*dot(d,s) + gc0*lc*inner(nabla_grad(d), nabla_grad(s)) - 2.0*hist_0*s)*dx(0) \
   + ((2.0*hist_1 + gc1/lc)*dot(d,s) + gc1*lc*inner(nabla_grad(d), nabla_grad(s)) - 2.0*hist_1*s)*dx(1)

Iu = inner(linearized_sigma_spectral_split(u, uold, dnew, lmbda0, mu0), epsilon_voigt(v))*dx(0) \
   + inner(linearized_sigma_spectral_split(u, uold, dnew, lmbda1, mu1), epsilon_voigt(v))*dx(1)

Ad, Ld = lhs(Id), rhs(Id)
Au, Lu = lhs(Iu), rhs(Iu)

u, d = Function(W), Function(V)

prob_dmge   = LinearVariationalProblem(Ad, Ld, d)
prob_disp   = LinearVariationalProblem(Au, Lu, u, bcs)

solver_dmge = LinearVariationalSolver(prob_dmge)
solver_disp = LinearVariationalSolver(prob_disp)
#----------------------------------------------------------------------#


#----------------------------------------------------------------------#
# Staggered algorithm
nsteps = 1000
delta  = 1E-4

vtkfile_d = File('results/damagefield.pvd')

d.vector().zero()
dnew.assign(d)

ud.t += delta

uold.assign(u)
solver_disp.solve()
unew.assign(u)

vtkfile_d << (d, 0)

for n in range(1,nsteps+1):
    ud.t += delta

    solver_dmge.solve()
    dnew.assign(d)

    vtkfile_d << (d, n)

    uold.assign(u)
    solver_disp.solve()
    unew.assign(u)
