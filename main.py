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

info(parameters, True)

#load mesh
mesh = Mesh('meshes/mesh_fenics.xml')
