from fenics import *

class fullDegradation:

    def __init__(self, lame, loading, mesh):
        self.lmbda  = lame[0]
        self.mu     = lame[1]
        self.nsteps = loading[0]
        self.delta  = loading[1]
