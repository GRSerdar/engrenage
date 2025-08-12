import numpy as np

from core.grid import Grid
from bssn.tensoralgebra import *
from bssn.bssnvars import *

# Constants for tensor algebra
one_two = 1.0/2.0
one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0
two_nine = 2.0/9.0
third_two = 3.0/2.0

def compute_L_GB(bssn_vars, bssn_rhs, d1, d2, grid, background):
    """
    Some extremely discriptive comment
    """
    r = grid.r
    N = grid.num_points

    ################### BSSN variables ################### 
    K = bssn_vars.K  
    phi = bssn_vars.phi  # Conformal factor (this is consistent with the pdf)

    em4phi = np.exp(-4.0* phi) 
    e4phi = 1/em4phi

    lapse = bssn_vars.lapse  
    ilapse = (lapse)**(-1) # inverse-lapse, i-lapse
    
    shift_U = bssn_vars.shift_U #scaled shift (lower case)
    Shift_U = background.inverse_scaling_vector * shift_U #Captial Shift 

    # Barred Metric and extrinsic curvature tensors
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    bar_A_LL = get_bar_A_LL(r, bssn_vars, background) 
    # bar_A_UU = get_bar_A_UU(r, bssn_vars, background) #[not used]

    #Derivative terms | notice that the derivative is only taken of the scaled variables.
    d1_phi = d1.phi  # Partial_i phi
    d2_phi = d2.phi  # Partial_i partial_j phi
    d1_K = d1.K  # Partial_i K
    d1_lapse = d1.lapse  # Partial_i alpha
    d2_lapse = d2.lapse  # Partial_i partial_j alpha

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

    ################### BSSN variables ################### 

    #________________________________________________________________________________________
    # Construction of Mij

    #barred ricci
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                              Delta_U, Delta_ULL, Delta_LLL, 
                                              bar_gamma_UU, bar_gamma_LL, background)

    # Writing out the Ricci tensor in terms of the barred ricci and barred christoffel symbol
    # (because it is a quantity that is already calculated)
    Rij = (bar_Rij 
           - 2*d2_phi
           + 2*np.einsum('xlij,xl->xij', bar_chris, d1_phi)
           - 2*np.einsum('xij,xlm,xlm->xij',bar_gamma_LL, bar_gamma_UU, d2_phi)
           + 2*np.einsum('xij, xlm, xklm, xk->xij', bar_gamma_LL, bar_gamma_UU, bar_chris, d1_phi)
           + 4*np.einsum('xi, xj->xij', d1_phi, d1_phi)
           - 4*np.einsum('xij, xlm, xl, xm->xij', bar_gamma_LL, bar_gamma_UU, d1_phi, d1_phi))
    
    #\bar{A}_ij \bar{A}_j^k
    AikAjk = np.einsum('xik, xkb, xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)

    # M_ij
    M_LL = (Rij 
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * bar_gamma_LL* K[:, np.newaxis, np.newaxis] * K[:, np.newaxis, np.newaxis]
                    + one_third * K[:, np.newaxis, np.newaxis] * bar_A_LL
                    - AikAjk))
    
    M_LL_sym = one_two * (M_LL + M_LL.swapaxes(1,2))

    #Trace M_ij
    Trace_M = get_trace(M_LL_sym, bar_gamma_UU)

    #Trace Free M_{ij}
    TraceFree_M_LL = M_LL_sym - one_third * bar_gamma_LL*Trace_M[:, np.newaxis, np.newaxis]
    
    #Trace Free M^{ij}
    TraceFree_M_UU = np.einsum('xik, xjl, xij->xkl',bar_gamma_UU, bar_gamma_UU, TraceFree_M_LL)

    #________________________________________________________________________________________
    # Construction of N_i

    #\bar{D}_j \bar{A}_i^j
    bar_D_bar_A_LU = (np.einsum('xjb, xjib->xi',bar_gamma_UU, a_times_d1_s)
                      + np.einsum('xjb, xjib->xi',bar_gamma_UU, s_times_d1_a)
                      - np.einsum('xjb, xlji,xlb->xi', bar_gamma_UU, bar_chris, bar_A_LL)
                      - np.einsum('xjb, xljb, xil->xi',bar_gamma_UU, bar_chris, bar_A_LL))
    
    gamma_A_LL_d1_phi = np.einsum('xjb, xib, xj->xi',bar_gamma_UU ,bar_A_LL, d1_phi)

    N_L = bar_D_bar_A_LU + 6*gamma_A_LL_d1_phi - two_thirds* d1_K

    N_U = np.einsum('xai, xi->xa', bar_gamma_UU, N_L)

    #N_iN^i
    N_squared = np.einsum('xi, xia, xa->x', N_L, bar_gamma_UU,N_L)
    #________________________________________________________________________________________
    # Line 1
    # Does not include advection!!!
    dKdt = bssn_rhs.K

    #D^iD_i \alpha    
    D2_lapse = (em4phi*(np.einsum('xia, xai->x',bar_gamma_UU, d2_lapse)
                        - np.einsum('xia, xkai, xk->x', bar_gamma_UU, bar_chris, d1_lapse)
                        - 2* np.einsum('xia, xa, xi->x', bar_gamma_UU, d1_lapse, d1_phi)
                        - 2* np.einsum('xia, xi, xa->x', bar_gamma_UU, d1_lapse, d1_phi)
                        + 6* np.einsum('xkl, xl, xk->x', bar_gamma_UU, d1_phi, d1_lapse)))
    
    #\bar{A}_{ij} \bar{A}^{ij}
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    line1 = lapse*(-four_thirds*Trace_M * (ilapse * dKdt 
                                     + ilapse * D2_lapse
                                     - Asquared 
                                     - one_third * K * K))

    #________________________________________________________________________________________
    #Line 2
     
    # Includes only the shift part of advection!!!
    dadt = bssn_rhs.a_LL

    # Removing the shift part of advection (for clean propagation)!!!
    bar_div_shift = get_bar_div_shift(r, bssn_vars, d1, background)  
    dadt_perp = dadt + (2.0/3.0) * bar_div_shift[:,np.newaxis,np.newaxis] * bssn_vars.a_LL

    #Term 5
    bar_A_LL_d_Shift_U_symmetric = one_two*(np.einsum('xjk, xlj->xkl',bar_A_LL,d1_Shift_U)
                                    + np.einsum('xjl, xkj->xkl',bar_A_LL,d1_Shift_U )) #ASSUMPTION: I made the indices after arrow symmetric [hence i changed their order myseflf]

    #Term 6
    DkDl_lapse = (d2_lapse
                  - np.einsum('xwkl,xw->xkl',bar_chris, d1_lapse)
                  - 2*np.einsum('xl, xk->xkl', d1_phi , d1_lapse) 
                  - 2*np.einsum('xk, xl->xkl', d1_phi , d1_lapse) 
                  + 2*np.einsum('xkl, xwd, xd, xw->xkl',bar_gamma_LL, bar_gamma_UU, d1_phi, d1_lapse))
    
    # symmetrized it befor use
    DkDl_lapse_sym = one_two * (DkDl_lapse + DkDl_lapse.swapaxes(1,2))

    line2_term_1 = 8  * e4phi * np.einsum('xij, xij->x', TraceFree_M_UU , dadt_perp)

    # line 2 term 2 is set to zero, since we want to add the advection terms afterwards (all at the same time)
    line2_term_2 = 0 * (16  * e4phi * np.einsum('xij, xij->x', TraceFree_M_UU , bar_A_LL_d_Shift_U_symmetric))
    line2_term_3 = 8  * np.einsum('xij, xij->x', TraceFree_M_UU , DkDl_lapse_sym)

    line2 = lapse*ilapse*(line2_term_1 + line2_term_2 + line2_term_3)

    #________________________________________________________________________________________
    #Line 3

    # gave it a letter 'r' extra on the end to not have the same name as something defined earlier in another file (you never know...)
    bar_div_shiftr =  np.einsum('xii->x', d1_Shift_U) 

    AkjAjl = np.einsum('xkj, xja, xal->xkl',bar_A_LL, bar_gamma_UU, bar_A_LL)

    line3_term1 = 8 * e4phi * np.einsum('xij, xij->x',TraceFree_M_UU, AkjAjl)

    line3_term2 = -two_thirds * 8 * e4phi * K *  np.einsum('xij, xij->x',TraceFree_M_UU, bar_A_LL)
    # line 3 term 3 is set to zero, since we want to add the advection terms afterwards (all at the same time)
    line3_term3 = 0 * (two_thirds * 8 * e4phi * ilapse * bar_div_shiftr * np.einsum('xij, xij->x',TraceFree_M_UU, bar_A_LL))

    line3 = lapse*(line3_term1 + line3_term2 + line3_term3)

    #________________________________________________________________________________________
    #Line 4   

    D_L_A_LL = (e4phi[:, np.newaxis, np.newaxis, np.newaxis] 
                * (4*np.einsum('xi, xjk->xijk',d1_phi,bar_A_LL) 
                   + a_times_d1_s + s_times_d1_a
                   - np.einsum('xlij, xlk-> xijk',bar_chris, bar_A_LL)
                   - 2*np.einsum('xj, xik->xijk',d1_phi, bar_A_LL) 
                   - 2*np.einsum('xi, xjk-> xijk',d1_phi, bar_A_LL)
                   + 2*np.einsum('xij, xlw, xw, xlk-> xijk', bar_gamma_LL, bar_gamma_UU, d1_phi, bar_A_LL)
                   - np.einsum('xlik, xjl-> xijk', bar_chris, bar_A_LL)
                   - 2*np.einsum('xk, xji-> xijk', d1_phi, bar_A_LL)
                   - 2*np.einsum('xi, xjk-> xijk', d1_phi, bar_A_LL)
                   + 2*np.einsum('xik, xlw, xw, xjl-> xijk', bar_gamma_LL, bar_gamma_UU, d1_phi, bar_A_LL))
                   )
    
    ##### CHECKUP #######

    ##### CHECKUP #######
    
    D_U_A_UU = (em4phi[:, np.newaxis, np.newaxis, np.newaxis])**3 *np.einsum('xjb, xkc, xia, xabc->xijk',bar_gamma_UU,bar_gamma_UU,bar_gamma_UU,D_L_A_LL)


    line4 = lapse*(-4*(2*np.einsum('xijk, xijk->x',D_L_A_LL, D_U_A_UU)
                - 2*np.einsum('xijk, xjik->x',D_L_A_LL, D_U_A_UU)
                - four_thirds* np.einsum('xi, xi->x',d1_K, N_U)
                - four_thirds*one_third* np.einsum('xi, xai ,xa->x',d1_K,bar_gamma_UU, d1_K )
                - 2*N_squared))

    L_GB = (line1 + line2 + line3 + line4)

    
    #print('line1: ', line1)
    #print('line2: ', line2)
    #print('line3: ', line3)
    #print('line4: ', line4)
    #print('_____________________')

    return L_GB