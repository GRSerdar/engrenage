import numpy as np
import matplotlib.pyplot as plt

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

Gaur = []
LLine1 = []
LLine2 = []
LLine3 = []
LLine4 = []

# Line 1 objects
dkdt_list = []
asquared_list = []
K_list = []

line1t1 = []
line1t2 = []
line1t3 = []
line1t4 = []

# Line 2 objects
dadt_list = []
DkDl_lapse_Katy_list = []

# Line 3
ak_list = []
klist = []


# Line 4 objects
D_difference = []
D_1list = []
D_2list = []
d1K_N_U = []
finalfour = []
finalfour2 = []

# Extra variables 
Trace_M_list = []
Hamilton = []



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
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background) 

    #Derivative terms | notice that the derivative is only taken of the scaled variables.
    d1_phi = d1.phi  # Partial_i phi
    d2_phi = d2.phi  # Partial_i partial_j phi
    d1_K = d1.K  # Partial_i K
    d1_lapse = d1.lapse  # Partial_i alpha
    d2_lapse = d2.lapse  # Partial_i partial_j alpha

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

    ################### BSSN variables ######################

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
    
    r_Rij = background.inverse_scaling_matrix * Rij

    
    # Katy her Rij
    '''Rij_katy = (- 2.0 * d2_phi
                         + 4.0 * np.einsum('xi,xj->xij', d1_phi, d1_phi)
                         + 2.0 * np.einsum('xkij,xk->xij', bar_chris, d1_phi)
                         + bar_Rij)'''
    
    #\bar{A}_ij \bar{A}_j^k

    # Start using the inverse scaled Aij
    AikAjk = np.einsum('xik, xkb, xjb->xij', bar_A_LL, bar_gamma_UU, bar_A_LL)
    r_AikAjk = background.inverse_scaling_matrix * AikAjk

    # Rescaled bar gammas
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, bssn_vars.h_LL, background)
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)

    # M_ij (implemented with inverse scaled Aij)
    '''
    M_LL = (Rij_katy  # In stead of Rij 
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * r_bar_gamma_LL* K[:, np.newaxis, np.newaxis] * K[:, np.newaxis, np.newaxis]
                    + one_third * K[:, np.newaxis, np.newaxis] * bssn_vars.a_LL #inverse scaled
                    - r_AikAjk)) # invers scaled'''
    
    r_M_LL = (r_Rij  # In stead of Rij 
           + e4phi[:, np.newaxis, np.newaxis] *(two_nine * r_bar_gamma_LL* K[:, np.newaxis, np.newaxis] * K[:, np.newaxis, np.newaxis]
                    + one_third * K[:, np.newaxis, np.newaxis] * bssn_vars.a_LL #inverse scaled
                    - r_AikAjk)) # invers scaled
    
    M_LL = background.scaling_matrix * r_M_LL


    # Trace M_ij
    # Trace_M = np.einsum("xij, xij->x",bar_gamma_UU, M_LL)
    Trace_M = get_trace(r_M_LL, r_bar_gamma_UU)

    # Adding to list 
    Trace_M_list.append(Trace_M)

    # Trace Free M_{ij}
    TraceFree_M_LL = M_LL - one_third * bar_gamma_LL * Trace_M[:, np.newaxis, np.newaxis]

    #TraceFree_M_LL = background.scaling_matrix * r_TraceFree_M_LL
    # Rescaled gamma bar trace
    #TraceFree_M_LL = M_LL - one_third * r_bar_gamma_LL * Trace_M[:, np.newaxis, np.newaxis]
    
    # Trace Free M^{ij}
    TraceFree_M_UU = np.einsum('xik, xjl, xij->xkl',bar_gamma_UU, bar_gamma_UU, TraceFree_M_LL)

    #TraceFree_M_UU = background.scaling_matrix * r_TraceFree_M_UU

    #________________________________________________________________________________________
    # Construction of N_i

    # \bar{D}_j \bar{A}_i^j
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
    dKdt_perp = bssn_rhs.K

    dkdt_list.append(dKdt_perp)

    #D^iD_i \alpha  
     
    #THESE DIFFERENT NOTATIONS ALL GIVE THE SAME NUMERICAL RESULTS!!!

    '''
    D2_lapse = (em4phi*(np.einsum('xia, xai->x',bar_gamma_UU, d2_lapse)
                        - np.einsum('xia, xkai, xk->x', bar_gamma_UU, bar_chris, d1_lapse)
                        - 2* np.einsum('xia, xa, xi->x', bar_gamma_UU, d1_lapse, d1_phi)
                        - 2* np.einsum('xia, xi, xa->x', bar_gamma_UU, d1_lapse, d1_phi)
                        + 6* np.einsum('xkl, xl, xk->x', bar_gamma_UU, d1_phi, d1_lapse))) '''
                        

    D2_lapse = (em4phi*(np.einsum('xia, xai->x',bar_gamma_UU, d2_lapse)
                        - np.einsum('xia, xkai, xk->x', bar_gamma_UU, bar_chris, d1_lapse)
                        + 2* np.einsum('xia, xa, xi->x', bar_gamma_UU, d1_lapse, d1_phi)))
    
    
    #\bar{A}_{ij} \bar{A}^{ij}
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    asquared_list.append(Asquared)
    
    line1 = -four_thirds*(Trace_M * dKdt_perp
                          + Trace_M*D2_lapse
                          - Trace_M*lapse* Asquared
                          - Trace_M*lapse* one_third * K*K)
    
    
    LLine1.append(line1)

    line1t1.append(-four_thirds*(Trace_M * dKdt_perp))
    line1t2.append(-four_thirds*(Trace_M*D2_lapse))
    line1t3.append((four_thirds*(Trace_M*lapse* Asquared)))
    line1t4.append((four_thirds*(Trace_M*lapse* one_third * K*K)))
    #________________________________________________________________________________________
    #Line 2
     
    # Includes only the shift part of advection!!!
    dadt_gb = bssn_rhs.a_LL

    # Removing the shift part of advection (for clean propagation)!!!
    # This is only used to calculate the Gaus bonnet term  

    bar_div_shift =  np.einsum('xii->x', d1_Shift_U) + np.einsum('xiij,xj->x', bar_chris, Shift_U)   
    #bar_div_shift = get_bar_div_shift(r, bssn_vars, d1, background) #[not the same divshift as in bssnrhs]
    
    dadt_perp = dadt_gb + (2.0/3.0) * bar_div_shift[:,np.newaxis,np.newaxis] * bssn_vars.a_LL

    #Term 5
    #bar_A_LL_d_Shift_U_symmetric = one_two*(np.einsum('xjk, xlj->xkl',bar_A_LL,d1_Shift_U)+ np.einsum('xjl, xkj->xkl',bar_A_LL,d1_Shift_U )) 

    #Term 6
    DkDl_lapse = (d2_lapse
                  - np.einsum('xwkl,xw->xkl',bar_chris, d1_lapse)
                  - 2*np.einsum('xl, xk->xkl', d1_phi , d1_lapse) 
                  - 2*np.einsum('xk, xl->xkl', d1_phi , d1_lapse) 
                  + 2*np.einsum('xkl, xwd, xd, xw->xkl',bar_gamma_LL, bar_gamma_UU, d1_phi, d1_lapse))
    
    # I don't get this term in my equations but this is what Katy wrote in bssnrhs.py
    DkDl_lapse_Katy = -(- d2.lapse
                      + np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                      + 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                      + 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))
    
    


    line2_term_1 =  8  * e4phi * np.einsum('xij, xij->x', TraceFree_M_UU , dadt_perp)

    # Modified line (just playing around): 
    #line2_term_1 =  12  * e4phi * np.einsum('xij, xij->x', TraceFree_M_UU , dadt_perp)

    #########################################################################################################
    '''## ## line 2 term 2 is set to zero, since we want to add the advection terms afterwards (all at the same time)'''
    bar_A_LL_d_Shift_U_symmetric = (np.einsum('xjk, xlj->xkl',bar_A_LL,d1_Shift_U)
                                    + np.einsum('xjl, xkj->xkl',bar_A_LL,d1_Shift_U )) 
    
    line2_term_2 = (8 * e4phi * np.einsum('xij, xij->x', TraceFree_M_UU , bar_A_LL_d_Shift_U_symmetric))
    #########################################################################################################

    line2_term_3 =  8  * np.einsum('xij, xij->x', TraceFree_M_UU , DkDl_lapse)
    
    dadt_list.append(line2_term_1)
    DkDl_lapse_Katy_list.append(line2_term_3)
    
    # Normally there should be a (lapse*ilapse)* in front but removed it for computational reasons 
    line2 = (line2_term_1  + line2_term_3)
    
    LLine2.append(line2)
    
    #________________________________________________________________________________________
    #Line 3

    # gave it a letter 'r' extra on the end to not have the same name as something defined earlier in another file (you never know...)

    AkjAjl = np.einsum('xkj, xja, xal->xkl',bar_A_LL, bar_gamma_UU, bar_A_LL)

    line3_term1 = 8 * e4phi * np.einsum('xij, xij->x',TraceFree_M_UU, AkjAjl)

    line3_term2 = -two_thirds * 8 * e4phi * K *  np.einsum('xij, xij->x',TraceFree_M_UU, bar_A_LL)
    
    ############################################################################################################
    #### line 3 term 3 is set to zero, since we want to add the advection terms afterwards (all at the same time)
    '''bar_div_shiftr =  np.einsum('xii->x', d1_Shift_U) 
    line3_term3 = 0 * (two_thirds * 8 * e4phi * ilapse * bar_div_shiftr * np.einsum('xij, xij->x',TraceFree_M_UU, bar_A_LL))'''
    ############################################################################################################

    ### DEBUG ###
    ak_list.append(line3_term1)
    klist.append(line3_term2)
    ### DEBUG ### 

    line3 = lapse*(line3_term1 + line3_term2 )

    LLine3.append(line3)

    #________________________________________________________________________________________
    #Line 4   

    D_L_A_LL = (e4phi[:, np.newaxis, np.newaxis, np.newaxis] 
                * (4*np.einsum('xi, xjk->xijk',d1_phi,bar_A_LL) 
                   + a_times_d1_s + s_times_d1_a
                   - np.einsum('xlij, xlk-> xijk',bar_chris, bar_A_LL)
                   - 2*np.einsum('xj, xik->xijk',d1_phi, bar_A_LL) 
                   - 4*np.einsum('xi, xjk-> xijk',d1_phi, bar_A_LL) # Changed to 4
                   + 2*np.einsum('xij, xlw, xw, xlk-> xijk', bar_gamma_LL, bar_gamma_UU, d1_phi, bar_A_LL)
                   - np.einsum('xlik, xjl-> xijk', bar_chris, bar_A_LL)
                   - 2*np.einsum('xk, xji-> xijk', d1_phi, bar_A_LL)
                   # - 2*np.einsum('xi, xjk-> xijk', d1_phi, bar_A_LL)
                   + 2*np.einsum('xik, xlw, xw, xjl-> xijk', bar_gamma_LL, bar_gamma_UU, d1_phi, bar_A_LL))
                   )
    
    ##### CHECKUP #######
   
    ##### CHECKUP #######
    
    D_U_A_UU = (em4phi[:, np.newaxis, np.newaxis, np.newaxis]
                * em4phi[:, np.newaxis, np.newaxis, np.newaxis]
                * em4phi[:, np.newaxis, np.newaxis, np.newaxis] 
                * np.einsum('xjb, xkc, xia, xabc->xijk',bar_gamma_UU,bar_gamma_UU,bar_gamma_UU,D_L_A_LL))
    

    line4 = lapse*(-4*(2*np.einsum('xijk, xijk->x',D_L_A_LL , D_U_A_UU)
                            - 2*np.einsum('xijk, xjik->x',D_L_A_LL , D_U_A_UU)
                            - four_thirds* np.einsum('xi, xi->x',d1_K, N_U)
                            - four_thirds*one_third* np.einsum('xi, xai ,xa->x',d1_K,bar_gamma_UU, d1_K )
                            - 2 * N_squared))
    
    #### DEBUG ################################################################
    D_LLL_D_UUU = -4*(2*np.einsum('xijk, xijk->x',D_L_A_LL , D_U_A_UU) - 2*np.einsum('xijk, xjik->x',D_L_A_LL , D_U_A_UU))
    D_1 = 2*np.einsum('xijk, xijk->x',D_L_A_LL , D_U_A_UU)
    D_2 = 2*np.einsum('xijk, xjik->x',D_L_A_LL , D_U_A_UU)
    lastterm = 4* four_thirds*one_third* np.einsum('xi, xai ,xa->x',d1_K,bar_gamma_UU, d1_K )
    lastterm2 = 4* four_thirds* np.einsum('xi, xi->x',d1_K, N_U)

    D_difference.append(D_LLL_D_UUU)
    D_1list.append(D_1)
    D_2list.append(D_2)
    d1K_N_U.append(- 2 * N_squared)
    finalfour.append(lastterm)
    finalfour2.append(lastterm2)
    LLine4.append(line4)
    
    # Calculating the ham constraint to verify that trace of Mij is equal to it 
    bar_R = get_trace(bar_Rij, bar_gamma_UU)
    Ham = (two_thirds * K * K - Asquared
                      + em4phi * ( bar_R
                                   - 8.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1_phi, d1_phi)
                                   - 8.0 * np.einsum('xij,xij->x', bar_gamma_UU, d2_phi)
                                   + 8.0 * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1_phi)))
    #### DEBUG ################################################################
    
    L_GB = (line1 + line2 + line3 + line4)

    Hamilton.append(Ham)
    Gaur.append(L_GB)
    return L_GB