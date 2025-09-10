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
D_L_K_N_U_list = []
D_L_K_D_U_K_list = []

line1_ = []
line2_ = []
line3_ = []
line4_ = []

line1_1 = []
line1_2 = []
line1_3 = []
line1_4 = []

line2_1 = []
line2_2 = []
line2_3 = []
line2_4 = []

line3_1 = []
line3_2 = []
line3_3 = []
line3_4 = []


line4_1 = []
line4_2 = []
line4_3 = []
line4_4 = []

MTFsquared = [] 
MTFsquared2 = [] 
MTFsquared3 = []
MTFsquared4 = []
MTFsquared5 = []
MTFsquared6 = []
MTFsquared7 = []
MTFsquared8 = []

DiAjkDiAjk = []
DDalpha_DDalpha = []
D2lapser = []
Asqua = []
KKK = []
dkldkllapse = []
TFM = []
NS = []

RicciScalar = []
RicciScalar2  = []
barar = []
Contract = []


def compute_L_GB(bssn_vars, bssn_rhs, d1, d2, grid, background):
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
    
    '''
    # Rij from Katy
    Rij2 = (- 2.0 * d2.phi
           + 4.0 * np.einsum('xi,xj->xij', d1.phi, d1.phi)
           + 2.0 * np.einsum('xkij,xk->xij', bar_chris, d1.phi)
           + bar_Rij)
       '''
    
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
    '''TraceFree_M_LL = M_LL - one_third * bar_gamma_LL * Trace_M[:, np.newaxis, np.newaxis]'''
    TraceFree_M_LL = M_LL - one_third * gamma_LL * Trace_M[:, np.newaxis, np.newaxis]

    # TraceFree_M_UU
    '''TraceFree_M_UU = em4phi[:,np.newaxis,np.newaxis]*em4phi[:,np.newaxis,np.newaxis]*np.einsum("xia, xjb, xab->xij",bar_gamma_UU, bar_gamma_UU, TraceFree_M_LL)'''
    TraceFree_M_UU = np.einsum("xia, xjb, xab->xij",gamma_UU, gamma_UU, TraceFree_M_LL)

    # rescaled TraceFree_M_UU
    r_TraceFree_M_UU = background.scaling_matrix * TraceFree_M_UU

    # \bar{D}_j \bar{A}_i^j
    '''
    bar_D_A_LU = (  np.einsum("xjm, xjim->xi",bar_gamma_UU, a_times_d1_s)
                  + np.einsum("xjm, xjim->xi", bar_gamma_UU, s_times_d1_a)
                  - np.einsum("xjm, xkji, xkm->xi", bar_gamma_UU, bar_chris, bar_A_LL)
                  - np.einsum("xjm, xkjm, xik->xi", bar_gamma_UU, bar_chris, bar_A_LL))
       '''
    
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

    '''
    D2_lapse = em4phi*(  np.einsum("xai, xai->x",bar_gamma_UU, d2.lapse)
                       - np.einsum("xai, xkai, xk->x",bar_gamma_UU, bar_chris, d1.lapse)
                       - 2 * np.einsum("xai, xi, xa->x",bar_gamma_UU, d1.phi, d1.lapse)
                       + 2 * np.einsum("xij, xkl, xl, xk->x", bar_gamma_LL, bar_gamma_UU, d1.phi, d1.lapse))
       ''' #YOU SHOULD CHECK WHY BOTH DEFINITIONS ARE DIFFERENT.
    
    # D_i D^i lapse # The correct expression that I copied from bssnrhs.py
    D2_lapse = em4phi*(np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                  - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse)
                  + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi))


    # bar A_ij bar A^ij
    Asquared = get_bar_A_squared(r, bssn_vars, background)

    # First line in the gauss bonnet term
    #Line1 = -four_thirds*Trace_M*(dKdt_perp + D2_lapse + bssn_vars.lapse * Asquared - one_third*bssn_vars.lapse * bssn_vars.K*bssn_vars.K)

    Line1 = (-four_thirds*Trace_M* dKdt_perp 
             -four_thirds*Trace_M* D2_lapse 
             -four_thirds*Trace_M* bssn_vars.lapse * Asquared 
             +four_thirds*Trace_M* one_third * bssn_vars.lapse * bssn_vars.K * bssn_vars.K)

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 2

    # Importing dadt from bssn_rhs (reusing it in stead ofecalculating it)
    dadt_perp = bssn_rhs.a_LL
    
    # D_k D_l lapse Correct
    DkDl_lapse = (d2.lapse
                  - np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                  - 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                  - 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))

    # There is a difference between divshift and just the one derivative of shift...
    Line2 = (8*e4phi * np.einsum("xkl, xkl->x", TraceFree_M_UU, background.scaling_matrix * dadt_perp) 
            + 8* e4phi * np.einsum("xkl, xjk, xlj->x",TraceFree_M_UU, bar_A_LL,d1_Shift_U)
            + 8* e4phi * np.einsum("xkl, xjl, xkj->x", TraceFree_M_UU, bar_A_LL, d1_Shift_U)
            + 8* np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 3
    
    div_shift =  np.einsum('xii->x', d1_Shift_U)+ np.einsum('xiij,xj->x', bar_chris, Shift_U)

    Line3 = 8*e4phi * (bssn_vars.lapse *np.einsum("xkl, xmj, xkj, xml->x",TraceFree_M_UU, bar_gamma_UU, bar_A_LL, bar_A_LL)
                       -two_thirds*(bssn_vars.lapse * bssn_vars.K -div_shift)*np.einsum("xij,xij->x",TraceFree_M_UU,bar_A_LL))

    ###########################################################################################################################
    ###########################################################################################################################
    # Construction of line 4

    # D_i A_jk

    '''
    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis] *(s_times_d1_a + a_times_d1_s
                      - np.einsum("xkab, xkc->xabc",bar_chris, bar_A_LL)
                      - np.einsum("xkac, xbk->xabc",bar_chris, bar_A_LL)
                      - 2*np.einsum("xb, xac->xabc",d1.phi, bar_A_LL) 
                      + 2 *np.einsum("xab, xkl, xl, xkc->xabc",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL )
                      - 2 *np.einsum("xc, xba->xabc",d1.phi, bar_A_LL)
                      + 2 *np.einsum("xac, xkl, xl, xbk->xabc",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))'''
    

    # Correct
    '''
    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis]*(+ a_times_d1_s + s_times_d1_a
                                                             - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                                                             - np.einsum("xmik, xjm->xijk",bar_chris, bar_A_LL)
                                                             - 2*np.einsum("xj, xik->xijk",d1.phi, bar_A_LL)
                                                             - 2*np.einsum("xk, xji->xijk",d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xij, xml, xl, xmk->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xik, xml, xl, xjm->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))
      '''
   
    D_L_A_LL = e4phi[:, np.newaxis, np.newaxis, np.newaxis]*(np.einsum("xijk->xijk",a_times_d1_s)
                                                             + np.einsum("xijk->xijk",s_times_d1_a)
                                                             - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                                                             - np.einsum("xmik, xjm->xijk",bar_chris, bar_A_LL)
                                                             - 2*np.einsum("xj, xik->xijk",d1.phi, bar_A_LL)
                                                             - 2*np.einsum("xk, xji->xijk",d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xij, xml, xl, xmk->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL)
                                                             + 2*np.einsum("xik, xml, xl, xjm->xijk",bar_gamma_LL, bar_gamma_UU, d1.phi, bar_A_LL))

    # D^i A^jk
    D_U_A_UU = ((np.einsum("xai, xbj, xkc, xabc->xijk",gamma_UU, gamma_UU, gamma_UU, D_L_A_LL)))

    # Contraction 
    DiAjkDiAjk_ = np.einsum('xijk,xijk->x', D_L_A_LL, D_U_A_UU)
    
    # D_i A_jk * D^[j A^j]k
    Product = ( np.einsum("xijk, xijk->x", D_L_A_LL, D_U_A_UU) - np.einsum("xijk, xjik->x", D_L_A_LL, D_U_A_UU))
    
    # D_i K N^i
    # D_L_K_N_U = em4phi * np.einsum("xai, xi, xa->x",bar_gamma_UU,d1.K, N_L)
    D_L_K_N_U = np.einsum("xai, xi, xa->x",bar_gamma_UU,d1.K, N_L)

    # D_i K D_i K
    D_L_K_D_U_K = np.einsum("xai, xi, xa->x", bar_gamma_UU, d1.K, d1.K)
    
    # Fourth line in the Gauss Bonnet term

    # put a (-1) in front due to a suspected minus sign error in the equations  + a factor of e4phi ?
    Line4 =  ((-4)*bssn_vars.lapse*2*Product 
             + (4)*bssn_vars.lapse*four_thirds * D_L_K_N_U 
             + (4)*bssn_vars.lapse*four_thirds * one_third * D_L_K_D_U_K 
             + (4)*bssn_vars.lapse*2*N_squared)
    
    ###########################################################################################################################
    ##########################################################################################################################
    # Playground For mathematica vs pyhton...

    DiAjkDiAjk.append((np.einsum("xijk, xijk->x", D_L_A_LL, D_U_A_UU)))
    #D2lapser.append(D2_lapse)
    
    DDalpha_DDalpha.append(np.einsum("xij, xij->x", DkDl_lapse, DkDl_lapse))
    Asqua.append(Asquared)
    
    D2lapser.append(D2_lapse)
    KKK.append(D_L_K_D_U_K)

    dkldkllapse.append((np.einsum("xij, xai, xbj, xab->x",DkDl_lapse,gamma_UU,gamma_UU, DkDl_lapse)))
    TFM.append(np.einsum("xij, xij->x", TraceFree_M_LL, TraceFree_M_UU))

    NS.append(N_squared)
    RicciScalar.append(np.einsum("xij, xij->x",Rij,gamma_UU))
    # RicciScalar2.append(np.einsum("xij, xij->x",Rij2,gamma_UU))

    bar_D_L_bar_A_LL = (a_times_d1_s + s_times_d1_a
                        - np.einsum("xmij, xmk->xijk",bar_chris, bar_A_LL)
                        - np.einsum("xmik, xjm->xikj",bar_chris, bar_A_LL))
    
    U_bar_D_L_bar_A_LL = (np.einsum("xai, xbj, xkc, xijk->xabc",bar_gamma_UU, bar_gamma_UU, bar_gamma_UU,bar_D_L_bar_A_LL))

    contract = np.einsum("xijk, xijk->x",bar_D_L_bar_A_LL, U_bar_D_L_bar_A_LL)
    Contract.append(contract)

    

    ###########################################################################################################################
    ##########################################################################################################################
    # Lists to store variables for plotting
    line1_.append(Line1)
    line2_.append(Line2)
    line3_.append(Line3)
    line4_.append(Line4)

    line1_1.append(-four_thirds*Trace_M* dKdt_perp)
    line1_2.append(-four_thirds*Trace_M* D2_lapse)
    line1_3.append(-four_thirds*Trace_M* bssn_vars.lapse * Asquared)
    line1_4.append(+four_thirds*Trace_M* one_third*bssn_vars.lapse * bssn_vars.K*bssn_vars.K)

    line2_1.append(8*e4phi * np.einsum("xkl, xkl->x", r_TraceFree_M_UU, dadt_perp)) ###
    line2_2.append(8* e4phi * np.einsum("xkl, xjk, xlj->x",TraceFree_M_UU, bar_A_LL,d1_Shift_U))
    line2_3.append(8* e4phi * np.einsum("xkl, xjl, xkj->x", TraceFree_M_UU, bar_A_LL, d1_Shift_U))
    line2_4.append(8* np.einsum("xkl, xkl->x", TraceFree_M_UU, DkDl_lapse))

    line3_1.append(8*e4phi * bssn_vars.lapse * np.einsum("xkl, xmj, xkj, xml->x",TraceFree_M_UU, bar_gamma_UU, bar_A_LL, bar_A_LL))
    line3_2.append(-8*e4phi * bssn_vars.lapse * two_thirds * bssn_vars.K * np.einsum("xkl, xkl->x",TraceFree_M_UU, bar_A_LL))
    line3_3.append(8*e4phi * two_thirds * np.einsum("xkl, xjj, xkl->x",TraceFree_M_UU, d1_Shift_U, bar_A_LL))

    # line4_1.append(((-4) * 2 * np.einsum("xijk, xijk->x",D_L_A_LL, D_U_A_UU) + (4) * 2 * np.einsum("xijk, xjik->x",D_L_A_LL, D_U_A_UU)))
    line4_1.append(((-4)*bssn_vars.lapse*2*Product))
    line4_2.append(((4)*bssn_vars.lapse*four_thirds * D_L_K_N_U ))
    line4_3.append(((4)*bssn_vars.lapse*four_thirds * one_third * D_L_K_D_U_K ))
    line4_4.append(((4)*bssn_vars.lapse*2*N_squared))

    MTFsquared.append(np.einsum('xij, xij->x', TraceFree_M_LL, TraceFree_M_UU))
    MTFsquared2.append(np.einsum('xij, xij->x', TraceFree_M_LL, bar_A_UU))
    MTFsquared3.append(np.einsum('xij, xij->x', background.inverse_scaling_matrix * TraceFree_M_LL, bssn_vars.a_LL))
    
    MTFsquared4.append(em4phi * np.einsum('xij, xij->x', background.inverse_scaling_matrix * TraceFree_M_LL, dadt_perp))
    MTFsquared5.append(em4phi * np.einsum('xij, xij->x', TraceFree_M_LL, dadt_perp))
    MTFsquared6.append(em4phi * np.einsum('xij, xij->x', TraceFree_M_LL, background.scaling_matrix * dadt_perp))
    MTFsquared7.append(np.einsum('xij, xij->x', TraceFree_M_LL, DkDl_lapse))
    MTFsquared8.append(np.einsum('xij, xij->x', background.inverse_scaling_matrix * TraceFree_M_LL, DkDl_lapse))

    ###########################################################################################################################
    ##########################################################################################################################
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
    Ennn.append(N_U) #Note that the upper index N is equivalent to the momentum constraint.
    barar.append(bar_R)

    # Does not give zero, but rather a 10^-17 value
    #print(np.min(np.einsum("xij, xij->x",TraceFree_M_UU, gamma_LL)))

    L_GB = Line1 + Line2 + Line3 + Line4
    Gaur.append(L_GB)

    return L_GB