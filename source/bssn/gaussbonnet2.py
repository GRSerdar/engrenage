import numpy as np
import matplotlib.pyplot as plt

from core.grid import Grid
from bssn.tensoralgebra import *
from bssn.bssnvars import *

one_two = 1.0/2.0
one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0
two_nine = 2.0/9.0
third_two = 3.0/2.0

Gaur = []
Hamilton = []
Momentum = []
Trace_M_list = []
Ennn = []

def compute_L_GB(bssn_vars, bssn_rhs, d1, d2, grid, background):
    """
    Some extremely discriptive comment
    """
    r = grid.r
    N = grid.num_points

    ################### BSSN variables ######################################
    em4phi = np.exp(-4.0* bssn_vars.phi) 
    e4phi = 1/em4phi
    
    shift_U = bssn_vars.shift_U #scaled shift (lower case)
    Shift_U = background.inverse_scaling_vector * shift_U #Captial Shift 

    # Barred Metric and extrinsic curvature tensors
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Rescaled bar gammas
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, bssn_vars.h_LL, background)
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)

    bar_A_LL = get_bar_A_LL(r, bssn_vars, background) 
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background) 

    # This is the same code reused from bssnrhs.py
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]  
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis]) #partial_i Shift_U
    
    d2_Shift_U = (np.einsum('xijk,xi->xijk', background.d2_inverse_scaling_vector, bssn_vars.shift_U)
         + np.einsum('xik,xij->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
         + np.einsum('xij,xik->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
         + np.einsum('xi,xijk->xijk', background.inverse_scaling_vector, d2.shift_U)) #partial_i partial_j Shift_U

    s_times_d1_a = background.scaling_matrix[:,:,:,np.newaxis] * d1.a_LL
    a_times_d1_s = bssn_vars.a_LL[:,:,:,np.newaxis] * background.d1_scaling_matrix

    # Compute connections
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background) #\bar \christoffel (that you will use basicly everywhere)

    ################### BSSN variables #########################################

    '''
    Rules for restarting:

     - Reuse as many objects that already exists
     - Don't make new variables, if not needed lapse == bssn_vars.lapse (use consistency across the code)
     - Calculate traces with the trace function (that already exists)
     - V_ij = s_ij v_ij     ||   v_ij = (s_ij)^-1 V_ij
     - N^i  = (s_i)^-1 n^i  ||   n^i  = s_i N^i 
     
     Importing dadt (which is initially scaled), should be unscaled before using it in the equations.
     To obtain the physical \partial_\perp \bar{A}_{ij}
     '''
    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of usefull objects

    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                              Delta_U, Delta_ULL, Delta_LLL, 
                                              bar_gamma_UU, bar_gamma_LL, background)
    
    '''Rij = (- 2.0 * d2.phi
           + 4.0 * np.einsum('xi,xj->xij', d1.phi, d1.phi)
           + 2.0 * np.einsum('xkij,xk->xij', bar_chris, d1.phi)
           + bar_Rij)'''
    
    Rij = (bar_Rij 
           - 2*d2.phi
           + 2*np.einsum('xlij,xl->xij', bar_chris, d1.phi)
           - 2*np.einsum('xij,xlm,xlm->xij',bar_gamma_LL, bar_gamma_UU, d2.phi)
           + 2*np.einsum('xij, xlm, xklm, xk->xij', bar_gamma_LL, bar_gamma_UU, bar_chris, d1.phi)
           + 4*np.einsum('xi, xj->xij', d1.phi, d1.phi)
           - 4*np.einsum('xij, xlm, xl, xm->xij', bar_gamma_LL, bar_gamma_UU, d1.phi, d1.phi))
    
    # \bar{A}_ij \bar{A}_j^k
    AikAjk = np.einsum('xik, xkb, xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)

    # M_ij
    M_LL = (Rij
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * bar_gamma_LL* bssn_vars.K[:, np.newaxis, np.newaxis] * bssn_vars.K[:, np.newaxis, np.newaxis]
                    + one_third * bssn_vars.K[:, np.newaxis, np.newaxis] * bar_A_LL - AikAjk)) 
    # M
    Trace_M = em4phi * get_trace(M_LL, bar_gamma_UU)

    # TraceFree_M_LL
    TraceFree_M_LL = M_LL - one_third * bar_gamma_LL * Trace_M[:, np.newaxis, np.newaxis]

    # TraceFree_M_UU
    TraceFree_M_UU = em4phi[:,np.newaxis,np.newaxis]*em4phi[:,np.newaxis,np.newaxis]*np.einsum("xia, xjb, xab->xij",bar_gamma_UU, bar_gamma_UU, TraceFree_M_LL)

    # \bar{D}_j \bar{A}_i^j
    bar_D_A_LU = (  np.einsum("xjm, xjim->xi",bar_gamma_UU, a_times_d1_s)
                  + np.einsum("xjm, xjim->xi", bar_gamma_UU, s_times_d1_a)
                  - np.einsum("xjm, xkji, xkm->xi", bar_gamma_UU, bar_chris, bar_A_LL)
                  - np.einsum("xjm, xkjm, xik->xi", bar_gamma_UU, bar_chris, bar_A_LL))
    
    # N_i 
    N_L = em4phi[:,np.newaxis]*(  bar_D_A_LU 
                                + 6 * np.einsum("xj, xjb, xib->xi",d1.phi, bar_gamma_UU, bar_A_LL)
                                - two_thirds * d1.K)
    
    # N^i
    N_U = em4phi[:, np.newaxis] * np.einsum("xij, xj->xi", bar_gamma_UU, N_L)

    # N_i N^i
    N_squared = em4phi* np.einsum('xi, xai, xa->x', N_L,bar_gamma_UU ,N_L)

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of Line 1 
    
    # Importing dKdt from bssn_rhs (reusing it in stead of recalculating it)
    dKdt_perp = bssn_rhs.K

    # D_i D^i lapse
    D2_lapse = em4phi*(  np.einsum("xai, xai->x",bar_gamma_UU, d2.lapse)
                       - np.einsum("xai, xkai, xk->x",bar_gamma_UU, bar_chris, d1.lapse)
                       - 2 * np.einsum("xai, xi, xa->x",bar_gamma_UU, d1.phi, d1.lapse)
                       + 2 * np.einsum("xij, xkl, xl, xk->x", bar_gamma_LL, bar_gamma_UU, d1.phi, d1.lapse))

    # bar A_ij bar A^ij
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    # First line in the gauss bonnet term
    Line1 = -four_thirds*Trace_M*(dKdt_perp + D2_lapse + bssn_vars.lapse * Asquared - one_third*bssn_vars.lapse * bssn_vars.K*bssn_vars.K)

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 2

    # Importing dadt from bssn_rhs (reusing it in stead ofecalculating it)
    dadt_perp =  bssn_rhs.a_LL

    # Advection term
    div_shift =  np.einsum('xii->x', d1_Shift_U)
    # div_shift += np.einsum('xiij,xj->x', bar_chris, Shift_U)

    # A_jk d_l beta^j + A_jl d_k beta^j (symmetric)
    A_LL_d_shift_symmetric = one_two * (np.einsum("xjk, xlj->xkl",bar_A_LL, d1_Shift_U) + np.einsum("xjl, xkj->xkl",bar_A_LL, d1_Shift_U))

    # D_k D_l lapse
    DkDl_lapse = (d2.lapse - np.einsum("xmkl, xm->xkl", bar_chris, d1.lapse)
                  - 2 * np.einsum("xl, xk->xkl", d1.phi, d1.lapse)
                  - 2 * np.einsum("xk, xl->xkl", d1.phi, d1.lapse)
                  + 2 * np.einsum("xkl, xmp, xp, xm->xkl", bar_gamma_LL,bar_gamma_UU, d1.phi, d1.lapse))

    # Second line in the Gauss Bonnet term
    Line2 = 8*(e4phi * np.einsum("xkl, xkl->x", TraceFree_M_UU, dadt_perp)
                + 2 * e4phi * np.einsum("xkl, xkl->x",TraceFree_M_UU,A_LL_d_shift_symmetric)
                + np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 3

    # bar A_kj bar A^j_l
    AkjAjl = np.einsum("xmj, xkj, xml->xkl",bar_gamma_UU, bar_A_LL, bar_A_LL)

    # Prefactor
    PreFac = bssn_vars.lapse * bssn_vars.K - div_shift

    # Third line in the Gauss Bonnet term
    Line3 = 8*(e4phi * bssn_vars.lapse * np.einsum("xkl, xkl->x",TraceFree_M_UU , AkjAjl)
               - two_thirds * e4phi * PreFac * np.einsum("xkl, xkl->x", TraceFree_M_UU, bar_A_LL))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 4

    # D_i A_jk
    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis] *(s_times_d1_a + a_times_d1_s
                      - np.einsum("xkab, xkc->xabc",bar_chris, bar_A_LL)
                      - np.einsum("xkac, xbk->xabc",bar_chris, bar_A_LL)
                      - 2*np.einsum("xb, xac->xabc",d1.phi, bar_A_LL) 
                      + 2 *np.einsum("xab, xkl, xl, xkc->xabc",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL )
                      - 2 *np.einsum("xc, xba->xabc",d1.phi, bar_A_LL)
                      + 2 *np.einsum("xac, xkl, xl, xbk->xabc",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))
    
    # D^i A^jk
    D_U_A_UU = (em4phi[:, np.newaxis, np.newaxis, np.newaxis] * em4phi[:, np.newaxis, np.newaxis, np.newaxis] * em4phi[:, np.newaxis, np.newaxis, np.newaxis]
                * np.einsum("xai, xbj, xkc, xabc->xijk",bar_gamma_UU, bar_gamma_UU, bar_gamma_UU, D_L_A_LL))
    
    # D_i A_jk * D^[j A^j]k
    Product = ( np.einsum("xijk, xijk->x", D_L_A_LL, D_U_A_UU) - np.einsum("xijk, xjik->x", D_L_A_LL, D_U_A_UU))
    
    # D_i K N^i
    D_L_K_N_U = np.einsum("xi, xi->x",d1.K, N_U)

    # D_i K D_i K
    D_L_K_D_U_K = em4phi * np.einsum("xai, xi, xa", bar_gamma_UU, d1.K, d1.K)
    
    # Fourth line in the Gauss Bonnet term
    Line4 = (-4)*bssn_vars.lapse*(2*Product 
                                  - four_thirds * D_L_K_N_U 
                                  - four_thirds * one_third * D_L_K_D_U_K 
                                  - 2*N_squared)

    ###########################################################################################################################
    ###########################################################################################################################


    ##### DEBUG ######

    # Calculating the ham constraint to verify that trace of Mij is equal to it.
    bar_R = get_trace(bar_Rij, bar_gamma_UU)

    Ham = (two_thirds * bssn_vars.K * bssn_vars.K - Asquared
                      + em4phi * ( bar_R
                                   - 8.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.phi, d1.phi)
                                   - 8.0 * np.einsum('xij,xij->x', bar_gamma_UU, d2.phi)
                                   + 8.0 * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.phi)))
    
    Mom = em4phi[:,np.newaxis] * (
                              np.einsum('xil,xjm,xlmj->xi', bar_gamma_UU, bar_gamma_UU, s_times_d1_a)
                            + np.einsum('xil,xjm,xlmj->xi', bar_gamma_UU, bar_gamma_UU, a_times_d1_s)
                            - np.einsum('xil,xjm,xnjl,xnm->xi', bar_gamma_UU, bar_gamma_UU, bar_chris, bar_A_LL)
                            - np.einsum('xil,xjm,xnjm,xln->xi', bar_gamma_UU, bar_gamma_UU, bar_chris, bar_A_LL)
                            + 6.0 * np.einsum('xij,xj->xi', bar_A_UU, d1.phi) 
                            - two_thirds * np.einsum('xij,xj->xi', bar_gamma_UU, d1.K))
    
    Hamilton.append(Ham)
    Momentum.append(Mom)
    Trace_M_list.append(Trace_M)
    Ennn.append(N_L)

    L_GB = Line1 + Line2 + Line3 + Line4
    Gaur.append(L_GB)

    return L_GB