from fenics import *

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, lambda_, mu):
    return lambda_*tr(epsilon(u))*Identity(2) + 2*mu*epsilon(u)

def energy_density(u, lambda_, mu):
    strain_tensor = 0.5*(grad(u) + grad(u).T)
    IC = tr(strain_tensor)
    ICC = tr(strain_tensor*strain_tensor)
    return (0.5*lambda_*IC**2) + mu*ICC

def epsilon_voigt(u):
    return as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1)+u[1].dx(0)])
