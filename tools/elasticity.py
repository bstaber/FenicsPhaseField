from fenics import *

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u, lambda_, mu, dim):
    return lambda_*tr(epsilon(u))*Identity(dim) + 2*mu*epsilon(u)

def epsilon_voigt(u):
    return as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1)+u[1].dx(0)])

def epsilon2voigt(e):
    return as_vector([e[0,0], e[1,1], 2.0*e[0,1]])

def sigma_voigt(u, lambda_, mu):
    c1 = lambda_ + 2.0*mu
    C = as_matrix([[c1, lambda_, 0.0], [lambda_, c1, 0.0], [0.0, 0.0, mu]])
    return C*epsilon_voigt(u)

def energy_density(u, lambda_, mu):
    strain_tensor = 0.5*(grad(u) + grad(u).T)
    IC = tr(strain_tensor)
    ICC = tr(strain_tensor*strain_tensor)
    return (0.5*lambda_*IC**2) + mu*ICC

def eigv_eigw_epsilon(e):
    tr_e  = tr(e)
    det_e = det(e)
    v1    = 0.5*tr_e + sqrt(tr_e*tr_e/4.0 - det_e)
    v2    = 0.5*tr_e - sqrt(tr_e*tr_e/4.0 - det_e)
    w1    = conditional(gt(abs(e[0,1]),1E-10), as_vector([v1-e[1,1], e[0,1]]), as_vector([1.0, 0.0]))
    w2    = conditional(gt(abs(e[0,1]),1E-10), as_vector([v2-e[1,1], e[0,1]]), as_vector([0.0, 1.0]))
    norm1 = sqrt(dot(w1,w1))
    norm2 = sqrt(dot(w2,w2))
    w1    = w1/norm1
    w2    = w2/norm2
    return v1, v2, w1, w2

def eigv_epsilon(e):
    tr_e  = tr(e)
    det_e = det(e)
    disc  = sqrt(tr_e*tr_e/4.0 - det_e)
    v1    = 0.5*tr_e + disc
    v2    = 0.5*tr_e - disc
    return v1, v2

def energy_density_positive(u, lambda_, mu):
    e     = epsilon(u)
    tr_e  = tr(e)
    det_e = det(e)
    disc  = sqrt(tr_e*tr_e/4.0 - det_e)
    v1    = 0.5*tr_e + disc
    v2    = 0.5*tr_e - disc
    v1p   = conditional(gt(v1,0.0),v1,0.0)
    v2p   = conditional(gt(v2,0.0),v2,0.0)
    return 0.5*lambda_*conditional(gt(tr_e,0.0),tr_e*tr_e,0.0) + mu*(v1p*v1p + v2p*v2p)

def sigma_spectral_split(u, dnew, lambda_, mu):
    e    = epsilon(u)
    tr_e = tr(e)
    v1, v2, w1, w2 = eigv_eigw_epsilon(e)

    ep = conditional(gt(v1,0.0),v1,0.0)*outer(w1,w1) + conditional(gt(v2,0.0),v2,0.0)*outer(w2,w2)
    en = conditional(lt(v1,0.0),v1,0.0)*outer(w1,w1) + conditional(lt(v2,0.0),v2,0.0)*outer(w2,w2)

    return ((1.0-dnew)*(1.0-dnew) + 1E-6)*(lambda_*conditional(gt(tr_e,0.0),tr_e,0.0)*Identity(2) + 2.0*mu*ep) \
                                         + lambda_*conditional(lt(tr_e,0.0),tr_e,0.0)*Identity(2) + 2.0*mu*en


def linearized_sigma_spectral_split(u, uold, dnew, lambda_, mu):
    enew    = epsilon(u)
    eold    = epsilon(uold)
    tr_eold = tr(eold)
    tr_enew = tr(enew)

    v1, v2, w1, w2 = eigv_eigw_epsilon(eold)

    m1 = as_vector([w1[0]*w1[0], w1[1]*w1[1], w1[0]*w1[1]])
    m2 = as_vector([w2[0]*w2[0], w2[1]*w2[1], w2[0]*w2[1]])

    GabplusGba = as_matrix([[ 4*m1[0]*m2[0],                  4*m1[2]*m2[2],                 2*m1[0]*m2[2] + 2*m1[2]*m2[0] ],
                            [ 4*m1[2]*m2[2],                  4*m1[1]*m2[1],                 2*m1[1]*m2[2] + 2*m1[2]*m2[1] ],
                            [ 2*m1[0]*m2[2] + 2*m1[2]*m2[0],  2*m1[1]*m2[2] + 2*m1[2]*m2[1], m1[0]*m2[1] + m1[1]*m2[0] + 2*m1[2]*m2[2] ]])

    tol = 1E-10

    Ep = conditional(gt(v1,0.0),1.0,0.0)*outer(m1,m1) + conditional(gt(v2,0.0),1.0,0.0)*outer(m2,m2) \
       + conditional(gt(abs(v1-v2),tol), (conditional(gt(v1,0.0),v1,0.0)-conditional(gt(v2,0.0),v2,0.0))*0.5*GabplusGba/(v1-v2), 0.5*GabplusGba)

    En = conditional(lt(v1,0.0),1.0,0.0)*outer(m1,m1) + conditional(lt(v2,0.0),1.0,0.0)*outer(m2,m2) \
       + conditional(gt(abs(v1-v2),tol), (conditional(lt(v1,0.0),v1,0.0)-conditional(lt(v2,0.0),v2,0.0))*0.5*GabplusGba/(v1-v2), 0.5*GabplusGba)

    enew_voigt    = epsilon2voigt(enew)
    IdentityVoigt = Constant((1.0,1.0,0.0))

    Rgt = conditional(gt(tr_eold,0.0),1.0,0.0)
    Rlt = conditional(lt(tr_eold,0.0),1.0,0.0)

    return ((1.0-dnew)*(1.0-dnew)+1E-6)*(0.5*lambda_*Rgt*tr_enew*IdentityVoigt + 2.0*mu*Ep*enew_voigt) \
                                       + 0.5*lambda_*Rlt*tr_enew*IdentityVoigt + 2.0*mu*En*enew_voigt
