import numpy as np

from core.grid import Grid
from bssn.tensoralgebra import *

one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0
two_nine = 2.0/9.0


def compute_L_GB(bssn_vars, d1, d2, rhs_dict, grid, background):
    r = grid.r
    N = grid.num_points

    # Variables and quantities
    K = bssn_vars.K                        # trace of extrinsic curvature
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)  # A_ij
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background)  # A^ij
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

    # Derivatives
    dK = d1.K                              # D_i K
    d1_alpha = d1.lapse                    # D_i alpha
    d2_alpha = d2.lapse                    # D_i D_j alpha
    d1_A_LL = d1.a_LL                      # D_k A_ij
    d2_A_LL = d2.a_LL                      # D_k D_l A_ij 

    # Lapse and conformal factor
    alpha = bssn_vars.lapse
    phi = bssn_vars.phi
    chi = np.exp(-4.0 * phi)
    chii = 1/chi #chii stands for chi inverse

    # Get bar_R_ij (Ricci tensor)
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    Delta_U, Delta_ULL, Delta_LLL = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                   Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL, background)
    
    #Define normal physical metric (up and down)
    gamma_UU = chii*bar_gamma_UU
    gamma_LL = chii*bar_gamma_LL

    # Construction M_ij from Eq. (5) 

    # \bar A_ik \bar A^k_j = gamma^kl A_ik A_jl
    AikAkj = get_AikAkj(bar_A_LL, bar_gamma_UU)

    M_LL = (bar_Rij 
            + chii*two_nine*bar_gamma_LL*K*K 
            + chii*one_third*K*bar_A_LL 
            - AikAkj)

    trace_M = get_trace(M_LL, gamma_UU)




    # Construction N_i from Eq. (6)

    bar_chris = get_bar_christoffel(r, Delta_ULL, background)

    # Divergence of A: \tilde{D}_j \bar{A}_i^j = \partial_j \bar{A}_{ij} - \bar{\Gamma}^k_{ji} \bar{A}_{kj}
    bar_div_A_L = np.einsum('xjik->xi', d1_A_LL) - np.einsum('xkjl,xlk->xj', bar_chris, bar_A_LL)

    # Gradient of chi from dphi: \partial_j \chi 
    d1_chi = -4.0 * chi[:, np.newaxis] * d1.phi  # shape (N, 3)

    # Now assemble N_L = \tilde{D}_j \bar{A}_i^j - (3/2χ) ∂_j χ - (2/3) ∂_i K
    N_L = (bar_div_A_L 
        - (3.0 / (2.0 * chi[:, np.newaxis])) * d1_chi 
        - (2.0 / 3.0) * dK)

    # Construction L^GB using expression in Eq. (4)
    L_GB = np.zeros(N)





    return L_GB