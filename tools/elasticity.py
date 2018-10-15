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

def epsilon2voigt(e):
    return as_vector([e[0,0], e[1,1], 2.0*e[0,1]])

def eigv_eigw_epsilon(e):
    tr_e  = tr(e)
    det_e = det(e)
    v1    = 0.5*tr_e + sqrt(tr_e*tr_e/4.0 - det_e)
    v2    = 0.5*tr_e - sqrt(tr_e*tr_e/4.0 - det_e)
    norm1 = sqrt(e[0,1]*e[0,1] + (v1-e[0,0])*(v1-e[0,0]))
    norm2 = sqrt(e[0,1]*e[0,1] + (v2-e[0,0])*(v2-e[0,0]))
    w1    = as_vector([e[0,1], v1-e[0,0]])/norm1
    w2    = as_vector([e[0,1], v2-e[0,0]])/norm2
    return v1, v2, w1, w2

def eigv_epsilon(e):
    tr_e  = tr(e)
    det_e = det(e)
    v1    = 0.5*tr_e + sqrt(tr_e*tr_e/4.0 - det_e)
    v2    = 0.5*tr_e - sqrt(tr_e*tr_e/4.0 - det_e)
    return v1, v2

def energy_density_positive(u, lambda_, mu):
    e      = epsilon(u)
    IC     = tr(e)
    v1, v2 = eigv_epsilon(e)
    return 0.5*lambda_*conditional(gt(IC,0.0),1.0,0.0)*IC**2 + 2.0*mu*(v1*v1 + v2*v2)

def sigma_spectral_split(u, uold, dnew, lambda_, mu):
    e     = epsilon(u)
    eold  = epsilon(uold)
    ICold = tr(eold)
    ICnew = tr(e)

    v1, v2, w1, w2 = eigv_eigw_epsilon(e)

    m1 = as_vector([w1[0]*w1[0], w1[1]*w1[1], w1[0]*w1[1]])
    m2 = as_vector([w2[0]*w2[0], w2[1]*w2[1], w2[0]*w2[1]])

    GabplusGba = as_matrix([[ 4*m1[0]*m2[0],                  4*m1[2]*m2[2],                 2*m1[0]*m2[2] + 2*m1[2]*m2[0] ],
                            [ 4*m1[2]*m2[2],                  4*m1[1]*m2[1],                 2*m1[1]*m2[2] + 2*m1[2]*m2[1] ],
                            [ 2*m1[0]*m2[2] + 2*m1[2]*m2[0],  2*m1[1]*m2[2] + 2*m1[2]*m2[1], m1[0]*m2[1] + m1[1]*m2[0] + 2*m1[2]*m2[2] ]])

    Ep = conditional(gt(v1,0.0),1.0,0.0)*outer(m1,m1) + conditional(gt(v2,0.0),1.0,0.0)*outer(m2,m2) \
       + conditional(gt(abs(v1-v2),1E-10), (conditional(gt(v1,0.0),v1,0.0)-conditional(gt(v2,0.0),v2,0.0) )*0.5*GabplusGba/(v1-v2), 0.5*GabplusGba)

    En = conditional(lt(v1,0.0),1.0,0.0)*outer(m1,m1) + conditional(lt(v2,0.0),1.0,0.0)*outer(m2,m2) \
       + conditional(gt(abs(v1-v2),1E-10), (conditional(lt(v1,0.0),v1,0.0)-conditional(lt(v2,0.0),v2,0.0))*0.5*GabplusGba/(v1-v2), 0.5*GabplusGba)

    eVoigt = epsilon2voigt(e)

    IdentityVoigt = as_vector([1.0, 1.0, 0.0])

    return ((1.0-dnew)*(1.0-dnew)+1E-6)*(0.5*lambda_*conditional(gt(ICold,0.0),1.0,0.0)*IdentityVoigt + 2.0*mu*Ep*eVoigt) \
           + 0.5*lambda_*conditional(lt(ICold,0.0),1.0,0.0)*IdentityVoigt + 2.0*mu*En*eVoigt
