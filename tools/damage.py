from fenics import *

def degradation(d):
    (1.0-d)*(1.0-d) + 1E-6
