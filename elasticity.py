from fenics import *

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, lambda_, mu):
    return lambda_*tr(epsilon(u))*Identity(2) + 2*mu*epsilon(u)
