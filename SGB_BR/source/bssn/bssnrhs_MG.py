# bssnrhs.py
# as in Etienne https://arxiv.org/abs/1712.07658v2
# see also Baumgarte https://arxiv.org/abs/1211.6632 for the eqns with matter

import numpy as np
from bssn.tensoralgebra import *
from bssn.bssnvars import *

# phi is the (exponential) conformal factor, that is \gamma_ij = e^{4\phi} \bar\gamma_{ij}
def get_bssn_rhs(bssn_rhs, r, matter, bssn_vars, d1, d2, grid, background, gb, gauge_coefficients):
    ####################################################################################################
    # Get all the useful quantities that will be used in the rhs

    # Constant factors needed in Modified Gauge (for modified gravity)
    # In the case you want to work in standard GR, just set these to zero.
    a, b = gauge_coefficients

    em4phi = np.exp(-4.0*bssn_vars.phi)
    e4phi  = 1.0/em4phi

    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    gamma_UU = em4phi[:,np.newaxis,np.newaxis] * bar_gamma_UU
    gamma_LL = e4phi [:,np.newaxis,np.newaxis] * bar_gamma_LL

    # The rescaled connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)

    # \bar \Gamma^i_{jk}
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # rescaled shift in terms of scaling factors and bssn_vars.shift_U
    Shift_U = background.inverse_scaling_vector * bssn_vars.shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]
                  + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis])
    d2_Shift_U = (np.einsum('xijk,xi->xijk', background.d2_inverse_scaling_vector, bssn_vars.shift_U)
                  + np.einsum('xik,xij->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
                  + np.einsum('xij,xik->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
                  + np.einsum('xi,xijk->xijk', background.inverse_scaling_vector, d2.shift_U))

    # This is the conformal divergence of the shift \bar D_i \beta^i
    bar_div_shift  = np.einsum('xii->x', d1_Shift_U)
    bar_div_shift += np.einsum('xiij,xj->x', bar_chris, Shift_U)

    # Trace of \bar A_ij and \bar{A}_{ij}\bar{A}^{ij}
    trace_bar_A   = get_trace_bar_A(r, bssn_vars, background)
    bar_A_squared = get_bar_A_squared(r, bssn_vars, background)
    bar_A_LL      = get_bar_A_LL(r, bssn_vars, background)
    bar_A_UU      = get_bar_A_UU(r, bssn_vars, background)

    ####################################################################################################
    # Importing all modified gravity objects ###########################################################
    bar_L_GB = gb.bar_L_GB
    M_LL     = gb.M_LL
    N_L      = gb.N_L
    Trace_M  = gb.Trace_M  

    bar_TraceFree_S_GB_LL = gb.bar_TraceFree_S_GB_LL
    bar_S_GB            = gb.bar_S_GB
    Omega_LL            = gb.Omega_LL
    d1Lambdadu         = gb.d1Lambdadu

    ####################################################################################################
    # Importing EM tensor (which already holds the backreaction corrections in rho and S_L #############

    EMtensor = matter.get_emtensor(r, bssn_vars, d1, d2, bssn_rhs, grid, background, gb)

    u = matter.u
    v = matter.v

    d1_u = matter.d1_u
    d2_u = matter.d2_u
    
    ####################################################################################################
    # First the conformal factor phi
    
    # Calculate rhs
    dphidt = (- one_sixth * bssn_vars.lapse * bssn_vars.K 
              + one_sixth * bar_div_shift)

    bssn_rhs.phi = dphidt     

    ####################################################################################################        
    # h is the rescaled part of the deviation from the hat metric
    # that is, \bar \gamma_ij = \hat \gamma_ij + \epsilon_ij
    # h_ij is rescaled \epsilon_ij (factors of 1/r etc)     
    
    # This is \hat\gamma_jk \hat D_i shift^k 
    # (note Etienne paper notation ambiguity - this is NOT \hat D_i \beta_j)
    hat_D_shift_U = (
         np.einsum('xjk,xki->xij', background.hat_gamma_LL, d1_Shift_U)
         + np.einsum('xjk,xkil,xl->xij', background.hat_gamma_LL, background.hat_christoffel, Shift_U)
   )
    
    # Rescale quantities because we want change in h not epsilon
    r_hat_D_shift_U = background.inverse_scaling_matrix * hat_D_shift_U
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)
    
    # Need to get the scalar factor in the right array dimension
    scalar_factor = two_thirds * (bssn_vars.lapse * trace_bar_A - bar_div_shift)
    
    # Now sum the values
    dhdt = (scalar_factor[:,np.newaxis,np.newaxis] * r_bar_gamma_LL
            - 2.0 * bssn_vars.lapse[:,np.newaxis,np.newaxis] * bssn_vars.a_LL 
            + r_hat_D_shift_U + np.transpose(r_hat_D_shift_U, axes=(0,2,1)))

    bssn_rhs.h_LL = dhdt     

    ####################################################################################################    
    # K is the trace of the extrinsic curvature 
    # that is K_ij = A_ij + 1/3 \gamma_ij K
   
    # Calculate \bar D^k \bar D_k lapse
    bar_D2_lapse = (np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                  - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse))

    # Calculate rhs (BackReaction Correction of EsGB is added)
    dKdt = (bssn_vars.lapse * (one_third * bssn_vars.K * bssn_vars.K 
                               + bar_A_squared + 0.5 * eight_pi_G * (EMtensor.rho + EMtensor.S))
            - em4phi * (bar_D2_lapse 
                        + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi)))
    
    # Extra term to RHS due to Modified Gauge 
    dKdt += ((bssn_vars.lapse*b)/(4*(1+b))) * (Trace_M - 2 * eight_pi_G * EMtensor.rho)
    
    # We put this to zero atm
    #bssn_rhs.K = dKdt 

    ####################################################################################################    
    # a_ij is the rescaled version of the conformal, traceless part of the extrinsic curvature
    # that is A_ij =  e^{4\phi) \tilde A_ij
    # a_ij is rescaled \tilde A_ij (factors of 1/r etc)    
    
    # Ricci tensor
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                              Delta_U, Delta_ULL, Delta_LLL, 
                                              bar_gamma_UU, bar_gamma_LL, background)
    
    # \bar A_ik \bar A^k_j = gamma^kl A_ik A_jl
    AikAkj = np.einsum('xkl,xik,xlj->xij', bar_gamma_UU, bar_A_LL, bar_A_LL)
    
    # The trace free part of the evolution eqn for A_ij (BackReaction Correction of EsGB is added)
    dAdt_TF_part = (bssn_vars.lapse[:,np.newaxis,np.newaxis] * 
                        (- 2.0 * d2.phi
                         + 4.0 * np.einsum('xi,xj->xij', d1.phi, d1.phi)
                         + 2.0 * np.einsum('xkij,xk->xij', bar_chris, d1.phi)
                         + bar_Rij - eight_pi_G * (EMtensor.Sij ))
                      - d2.lapse
                      + np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                      + 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                      + 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))
    
    trace = get_trace(dAdt_TF_part, bar_gamma_UU)
    trace = trace[:,np.newaxis,np.newaxis]

    # Rescale quantities because we want change in a_ij not A_ij
    dadt_TF_part = background.inverse_scaling_matrix * dAdt_TF_part
    r_AikAkj = background.inverse_scaling_matrix * AikAkj
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)
    
    # Calculate rhs    
    dadt = ( - two_thirds * bar_div_shift[:,np.newaxis,np.newaxis] * bssn_vars.a_LL
             + bssn_vars.lapse[:,np.newaxis,np.newaxis] * (- 2.0 * r_AikAkj
                                                 + bssn_vars.K[:,np.newaxis,np.newaxis] * bssn_vars.a_LL)
             + em4phi[:,np.newaxis,np.newaxis] * (dadt_TF_part - bssn_vars.lapse[:,np.newaxis,np.newaxis] * eight_pi_G 
                                                  - one_third * trace * r_bar_gamma_LL))

    #bssn_rhs.a_LL = dadt   

    ####################################################################################################    
    # lambda^i is the rescaled version of the constrained quantity \Lambda^i = \Delta^i
    # Where \Delta^k = \bar\gamma^ij (\bar\Gamma^k_ij - \hat\Gamma^k_ij)  

    # \bar \gamma^jk \hat D_j \hat D_k shift^i
    hat_D2_shift = (  np.einsum('xjk,xijk->xi', bar_gamma_UU, d2_Shift_U)
                + np.einsum('xjk,xikl,xlj->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                + np.einsum('xjk,xijl,xlk->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                - np.einsum('xjk,xljk,xil->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                + np.einsum('xjk,xiklj,xl->xi', bar_gamma_UU, background.d1_hat_christoffel, Shift_U)
                + np.einsum('xjk,xijl,xlkm,xm->xi', bar_gamma_UU, background.hat_christoffel, 
                                                    background.hat_christoffel, Shift_U)
                - np.einsum('xjk,xljk,xilm,xm->xi', bar_gamma_UU, background.hat_christoffel, 
                                                    background.hat_christoffel, Shift_U))
    # This is \bar D^i (\bar D_j \beta^j) note the raised index of j
    # We can use that D_i V^i = 1/sqrt(detgamma) d_i [sqrt(detgamma) V^i]
    # And that we impose det(bargamma) = det(hatgamma) which we know the derivs for analytically
    bar_D_div_shift = (np.einsum('xij,xkjk->xi', bar_gamma_UU, d2_Shift_U)
                     + (0.5 / background.det_hat_gamma[:,np.newaxis] * 
                        np.einsum('xij,xkj,xk->xi', bar_gamma_UU, d1_Shift_U, background.d1_det_hat_gamma))
                     + (0.5 / background.det_hat_gamma[:,np.newaxis] * 
                        np.einsum('xij,xjk,xk->xi', bar_gamma_UU, background.d2_det_hat_gamma, Shift_U))
                     - (0.5 / background.det_hat_gamma[:,np.newaxis] / background.det_hat_gamma[:,np.newaxis] 
                        * np.einsum('xij,xj,xk,xk->xi', bar_gamma_UU, background.d1_det_hat_gamma, 
                                                        background.d1_det_hat_gamma, Shift_U)))

    # Calculate rhs
    dlambdadt = (hat_D2_shift 
                  + two_thirds * Delta_U * bar_div_shift[:,np.newaxis]
                  + one_third * bar_D_div_shift
                  - 2.0 * np.einsum('xij,xj->xi', bar_A_UU, d1.lapse)
                  + 12.0 * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_A_UU, d1.phi)
                  + 2.0 * bssn_vars.lapse[:,np.newaxis] * np.einsum('xjk,xijk->xi', bar_A_UU, Delta_ULL)
                  - four_thirds * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_gamma_UU, d1.K)
                  - 2.0 * eight_pi_G * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_gamma_UU, EMtensor.Si))
    
    dlambdadt -= ((2*bssn_vars.lapse[:,np.newaxis]*b)/(1+b))*(np.einsum("xij, xj->xi",bar_gamma_UU, N_L) - eight_pi_G * np.einsum("xij, xj->xi",bar_gamma_UU, EMtensor.Si))

    # Rescale because we want change in lambda not Lambda
    dlambdadt[:] *= background.scaling_vector
    
    bssn_rhs.lambda_U = dlambdadt


    ####################################################################################################
    # Construction of matrix to complete backreaction
    r = grid.r
    N = grid.num_points

    # Calculate usefull quantities
    Trace_Omega = get_trace(Omega_LL, gamma_UU)
    TraceFree_Omega_LL = Omega_LL - one_third * gamma_LL * Trace_Omega[:, np.newaxis, np.newaxis]
    TraceFree_Omega_UU = np.einsum("xia, xjb,xij->xab",gamma_UU,gamma_UU,TraceFree_Omega_LL)
    TraceFree_M_LL = M_LL - one_third * gamma_LL * Trace_M[:, np.newaxis, np.newaxis]
    TraceFree_M_UU = np.einsum("xia, xjb, xab->xij",gamma_UU, gamma_UU, TraceFree_M_LL)

    # Define dirac deltas
    delta_U_L = np.einsum("xim, xmk->xik",gamma_UU, gamma_LL)

    # Define all matrix components
    X_ij_UU = (np.einsum("xki, xlj->xijkl", delta_U_L, delta_U_L)*(1- (2*eight_pi_G/3)* Trace_Omega[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
               + 8*np.pi*(2*(np.einsum("xli, xjm, xmk->xijkl", delta_U_L, TraceFree_Omega_LL, gamma_UU) + np.einsum("xlj, xim, xmk->xijkl",delta_U_L, TraceFree_Omega_LL, gamma_UU))
                          - four_thirds * np.einsum("xij, xkl->xijkl", gamma_LL, TraceFree_Omega_UU)
                          - 64 * d1Lambdadu[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * d1Lambdadu[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] * np.einsum("xij, xkl-> xijkl",TraceFree_M_LL, TraceFree_M_UU)))
    
    Y_ij = (8*np.pi / 3) * em4phi[:,np.newaxis, np.newaxis] * (32* d1Lambdadu[:,np.newaxis,np.newaxis] * d1Lambdadu[:,np.newaxis,np.newaxis]* TraceFree_M_LL * Trace_M[:,np.newaxis,np.newaxis] - 2 * TraceFree_Omega_LL)

    X_K_UU = (8*np.pi / em4phi[:,np.newaxis, np.newaxis]) * (TraceFree_Omega_UU - 16 * d1Lambdadu[:,np.newaxis,np.newaxis] * d1Lambdadu[:,np.newaxis,np.newaxis] * Trace_M[:,np.newaxis,np.newaxis] * TraceFree_M_UU)

    Y_K = (1+(16*np.pi/3)*(4*d1Lambdadu*d1Lambdadu* Trace_M * Trace_M - Trace_Omega))

    X_Pi_UU = -(8/em4phi[:,np.newaxis, np.newaxis]) * d1Lambdadu[:,np.newaxis, np.newaxis] * TraceFree_M_UU

    Y_Pi = (four_thirds) * d1Lambdadu * Trace_M

    # the RHS 
    Z_A_LL = background.scaling_matrix * dadt - (bssn_vars.lapse[:, np.newaxis, np.newaxis] * em4phi[:,np.newaxis, np.newaxis] * 8*np.pi * (bar_TraceFree_S_GB_LL))
    
    DKDT = dKdt + 4 * np.pi * bssn_vars.lapse  * (bar_S_GB)
    
    # in this function I just added the bar_L_GB in the last line (which was forgotten previously)
    dvdt =  ((bssn_vars.lapse * bssn_vars.K * v 
                 + 2.0 * bssn_vars.lapse * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.phi, d1_u)
                 +       bssn_vars.lapse * em4phi * np.einsum('xij,xij->x', bar_gamma_UU, d2_u)
                 +                         em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1_u)
                 -       bssn_vars.lapse * em4phi * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1_u)) 
                 
                 - bssn_vars.lapse * matter.dVdu(u)

                 + bssn_vars.lapse * d1Lambdadu * bar_L_GB) 
    
    # Iteration 2 (STILL GIVES SINGULAR MATRIX ERROR)
    #M = np.zeros((N,4,4))
    #Z = np.zeros((N,4,1))
    d = 4
    M = np.zeros((N,d,d))
    Z = np.zeros((N,d,1))

    ir, it, ip = i_r, i_t, i_p

    # First row
    M[:,0,0] = X_ij_UU[:,ir,ir,ir,ir]
    M[:,0,1] = X_ij_UU[:,ir,ir,it,it] + X_ij_UU[:,ir,ir,ip,ip]
    M[:,0,2] = Y_ij[:,ir,ir]

    # Second row
    M[:,1,0] = X_ij_UU[:,it,it,ir,ir]
    M[:,1,1] = X_ij_UU[:,it,it,it,it] + X_ij_UU[:,it,it,ip,ip]
    M[:,1,2] = Y_ij[:, it, it]

    # third row
    M[:,2,0] = X_K_UU[:,ir,ir]
    M[:,2,1] = X_K_UU[:,it,it] + X_K_UU[:,ip,ip]
    M[:,2,2] = Y_K

    # fourth row
    M[:,3,0] = X_Pi_UU[:,ir,ir]
    M[:,3,1] = X_Pi_UU[:,it,it] + X_Pi_UU[:,ip,ip]
    M[:,3,2] = Y_Pi
    M[:,3,3] = 1.0

    print('M: ', M)

    # RHS
    Z[:,0,0] = Z_A_LL[:,ir,ir]
    Z[:,1,0] = Z_A_LL[:,it,it] 
    Z[:,2,0] = DKDT
    Z[:,3,0] = dvdt

    dU = np.linalg.solve(M, Z)[...,0]  # shape (N,4)

    A_LL = np.zeros((N,3,3))
    A_LL[:,ir,ir] = dU[:,0]
    A_LL[:,it,it] = dU[:,1]
    A_LL[:,ip,ip] = dU[:,1]          
    
    bssn_rhs.a_LL = background.inverse_scaling_matrix * A_LL
    bssn_rhs.K    = dU[:,2]
    dPidt         = dU[:,3]

    dudt =  bssn_vars.lapse * v 

    return (dudt, dPidt)

    ####################################################################################################
    # end of bssn rhs
    ####################################################################################################
