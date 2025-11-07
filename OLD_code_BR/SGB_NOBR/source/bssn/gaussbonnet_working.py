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

def compute_L_GB(bssn_vars, bssn_rhs, d1, d2, grid, background):
    r = grid.r
    N = grid.num_points
    
    ################### BSSN variables ######################################
    em4phi = np.exp(-4.0* bssn_vars.phi) 
    e4phi = 1/em4phi
    
    ilapse = 1/(bssn_vars.lapse)

    shift_U = bssn_vars.shift_U #scaled shift (lower case)
    Shift_U = background.inverse_scaling_vector * shift_U #Captial Shift 

    # Barred Metric and extrinsic curvature tensors
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Physical Metrics (since we will need a few times in this file)
    gamma_UU = em4phi[:,np.newaxis,np.newaxis] *bar_gamma_UU
    gamma_LL = e4phi[:,np.newaxis,np.newaxis] * bar_gamma_LL

    bar_A_LL = get_bar_A_LL(r, bssn_vars, background) 
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background) 

    # This is the same code reused from bssnrhs.py
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis]  
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis]) 

    s_times_d1_a = background.scaling_matrix[:,:,:,np.newaxis] * d1.a_LL
    a_times_d1_s = bssn_vars.a_LL[:,:,:,np.newaxis] * background.d1_scaling_matrix

    # To deal with the scaling matrix indices being jki in stead of ijk 
    a_times_d1_s = np.moveaxis(a_times_d1_s, 3, 1)   # xbca -> xabc
    s_times_d1_a = np.moveaxis(s_times_d1_a, 3, 1)   # xbca -> xabc

    # Compute connections
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_chris = get_bar_christoffel(r, Delta_ULL, background) #\bar \christoffel (that you will use basicly everywhere)

    ################### BSSN variables #########################################

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of usefull objects

    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                              Delta_U, Delta_ULL, Delta_LLL, 
                                              bar_gamma_UU, bar_gamma_LL, background)
    
    Rij = (bar_Rij 
           - 2*d2.phi
           + 2*np.einsum('xlij,xl->xij', bar_chris, d1.phi)
           - 2*np.einsum('xij,xlm,xlm->xij',bar_gamma_LL, bar_gamma_UU, d2.phi)
           + 2*np.einsum('xij, xlm, xklm, xk->xij', bar_gamma_LL, bar_gamma_UU, bar_chris, d1.phi)
           + 4*np.einsum('xi, xj->xij', d1.phi, d1.phi)
           - 4*np.einsum('xij, xlm, xl, xm->xij', bar_gamma_LL, bar_gamma_UU, d1.phi, d1.phi))
    

    # \bar{A}_ij \bar{A}_j^k
    AikAjk = np.einsum('xik, xkb, xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)

    # M_ij [ASK Llibert which gamma should be insinde ?]
    M_LL = (Rij
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * bar_gamma_LL* bssn_vars.K[:, np.newaxis, np.newaxis] * bssn_vars.K[:, np.newaxis, np.newaxis]
                    + one_third * bssn_vars.K[:, np.newaxis, np.newaxis] * bar_A_LL - AikAjk)) 

    # M
    Trace_M = get_trace(M_LL, gamma_UU)

    # TraceFree_M_LL #[Changed the bar_gamma_LL --> gamma_LL]
    TraceFree_M_LL = M_LL - one_third * gamma_LL * Trace_M[:, np.newaxis, np.newaxis]
    print("TraceFree_M_LL", TraceFree_M_LL)

    # TraceFree_M_UU
    TraceFree_M_UU = np.einsum("xia, xjb, xab->xij",gamma_UU, gamma_UU, TraceFree_M_LL)
    
    # N_i (This should not have a conformal factor in front)
    N_L = (np.einsum("xjm, xjim->xi",bar_gamma_UU, a_times_d1_s)
           + np.einsum("xjm, xjim->xi", bar_gamma_UU, s_times_d1_a)
           - np.einsum("xjm, xkji, xkm->xi", bar_gamma_UU, bar_chris, bar_A_LL)
           - np.einsum("xjm, xkjm, xik->xi", bar_gamma_UU, bar_chris, bar_A_LL) 
           + 6 * np.einsum("xj, xjb, xib->xi",d1.phi, bar_gamma_UU, bar_A_LL)
           - two_thirds * d1.K)

    # N^i (it is actuallyt he upper index N which corresponds to the momentum constraint)
    N_U = em4phi[:, np.newaxis] * np.einsum("xij, xj->xi", bar_gamma_UU, N_L)

    # N_i N^i
    N_squared = np.einsum('xi, xai, xa->x', N_L,gamma_UU ,N_L)

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of Line 1 
    
    # Importing dKdt from bssn_rhs (reusing it in stead of recalculating it)    
    dKdt_perp =  bssn_rhs.K 
    
    #dKdt_perp =  bssn_rhs.K + np.einsum("xi,xi->x",Shift_U,d1.K)

    # D_i D^i lapse (same as in bssnrhs.py)
    D2_lapse = em4phi*(np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                  - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
                  + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi))

    # bar A_ij bar A^ij
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    # First line in the gauss bonnet term
    Line1 = (-four_thirds*Trace_M* (ilapse * dKdt_perp 
                                    + ilapse * D2_lapse 
                                    + Asquared 
                                    - bssn_vars.K * bssn_vars.K))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 2

    # dAdt_perp 
    dAdt_perp = (background.scaling_matrix * bssn_rhs.a_LL) 
    
    # Covariant derivative of the Shift^i
    div_shift =  np.einsum('xii->x', d1_Shift_U)+ np.einsum('xiij,xj->x', bar_chris, Shift_U)

    
    # bar A_kj bar A^j_l
    bar_A_LL_A_UL = np.einsum("xaj,xkj,xal->xkl",bar_gamma_UU,bar_A_LL,bar_A_LL)

    # D_k D_l lapse 
    DkDl_lapse = (d2.lapse
                  - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                  - 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                  - 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))
    
    Line2 = (8*(ilapse*e4phi * np.einsum("xkl,xkl->x",TraceFree_M_UU, dAdt_perp)
                + ilapse * np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse)
                + e4phi*(np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL_A_UL)
                         -two_thirds*(bssn_vars.K - ilapse * div_shift)*np.einsum("xkl,xkl->x",TraceFree_M_UU,bar_A_LL))))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 3

    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis]*(  a_times_d1_s
                                                             + s_times_d1_a
                                                             - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                                                             - np.einsum("xmik, xjm->xijk",bar_chris, bar_A_LL)
                                                             - 2*np.einsum("xj, xik->xijk",d1.phi, bar_A_LL)
                                                             - 2*np.einsum("xk, xji->xijk",d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xij, xml, xl, xmk->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xik, xml, xl, xjm->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))
    
    # D^i A^jk
    D_U_A_UU = ((np.einsum("xai, xbj, xkc, xabc->xijk",gamma_UU, gamma_UU, gamma_UU, D_L_A_LL)))

    # D_i A_jk * D^[j A^i]k
    Product = ( np.einsum("xijk, xijk->x", D_L_A_LL, D_U_A_UU) - np.einsum("xijk, xjik->x", D_L_A_LL, D_U_A_UU))

    # D_i K N^i
    D_L_K_N_U = np.einsum("xai, xi, xa->x",gamma_UU,d1.K, N_L)

    # D_i K D_i K
    D_L_K_D_U_K = np.einsum("xai, xi, xa->x", gamma_UU, d1.K, d1.K)

    Line3 = -4*(2*(Product)-four_thirds*D_L_K_N_U - four_thirds*one_third*D_L_K_D_U_K-2*N_squared)
    
    L_GB = Line1 + Line2 + Line3
    Gaur.append(L_GB)

    ### DEBUG ###
    # Extract the five spherical-symmetry blocks used in Mathematica:
    '''
    ir, it, ip = i_r, i_t, i_p

    DrArr     = D_L_A_LL[:, ir, ir, ir];   DrArrUUU = D_U_A_UU[:, ir, ir, ir]
    DrAtt     = D_L_A_LL[:, ir, it, it];   DrAttUUU = D_U_A_UU[:, ir, it, it]
    DrApp     = D_L_A_LL[:, ir, ip, ip];   DrAppUUU = D_U_A_UU[:, ir, ip, ip]
    DtArt     = D_L_A_LL[:, it, ir, it];   DtArtUUU = D_U_A_UU[:, it, ir, it]
    DpArp     = D_L_A_LL[:, ip, ir, ip];   DpArpUUU = D_U_A_UU[:, ip, ir, ip]

    ContractDrArr.append(DrArr*DrArrUUU)
    ContractDrAtt.append(DrAtt*DrAttUUU)
    ContractDrApp.append(DrApp*DrAppUUU)
    ContractDtArt.append(DtArt*DtArtUUU)
    ContractDpArp.append(DpArp*DpArpUUU)  

    Contract5_.append(DtArt*DrAttUUU)
    Contract6_.append(DrAtt*DtArtUUU)
    Contract7_.append(DpArp*DrAppUUU)
    Contract8_.append(DrApp*DpArpUUU)

    De = a_times_d1_s + s_times_d1_a
    Deriv.append(s_times_d1_a[:, it, it,ir])
    Deriv2.append(a_times_d1_s[:, it, it,ir])

    barAtt.append(bssn_vars.a_LL[:, it, it])
    barApp.append(bar_A_LL[:, ip, ip])
    '''
    ### DEBUG ###
    return L_GB

    ###########################################################################################################################
    ###########################################################################################################################