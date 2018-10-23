import matplotlib.pyplot as plt

from fenics import *
from tools.elasticity import *

#----------------------------------------------------------------------#
# Set some Fenics parameters
parameters["form_compiler"]["optimize"]       = True
parameters["form_compiler"]["cpp_optimize"]   = True
parameters["form_compiler"]["representation"] = "uflacs"
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
# Variational problems


#----------------------------------------------------------------------#
